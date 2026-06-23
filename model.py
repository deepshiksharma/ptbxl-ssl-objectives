import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(output_size)
        self.mp = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


def bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None):
    layers = []

    if bn:
        layers.append(nn.BatchNorm1d(n_in))

    if p != 0:
        layers.append(nn.Dropout(p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None:
        layers.append(actn)

    return layers


def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn_final=False, bn=True, act="relu", concat_pooling=True):
    if lin_ftrs is None:
        lin_ftrs = [2 * nf if concat_pooling else nf, nc]
    else:
        lin_ftrs = [2 * nf if concat_pooling else nf] + list(lin_ftrs) + [nc]

    if not isinstance(ps, (list, tuple)):
        ps = [ps]

    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps

    if act == "relu":
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    elif act == "elu":
        actns = [nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    else:
        raise ValueError(f"Unsupported head activation: {act}")

    layers = [AdaptiveConcatPool1d()] if concat_pooling else [nn.AdaptiveMaxPool1d(1)]
    layers.append(Flatten())

    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn=bn, p=p, actn=actn)

    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))

    return nn.Sequential(*layers)


class ConvLayer(nn.Sequential):
    def __init__(self, ni, nf, ks=3, stride=1, padding=None, bias=None, act_cls=nn.ReLU, init=nn.init.kaiming_normal_):
        if padding is None:
            padding = (ks - 1) // 2

        if bias is None:
            bias = False
        
        conv = nn.Conv1d(ni, nf, kernel_size=ks, stride=stride, padding=padding, bias=bias)

        if init is not None:
            init(conv.weight)

        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0.0)

        layers = [conv, nn.BatchNorm1d(nf)]

        if act_cls is not None:
            layers.append(act_cls())

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, expansion, ni, nf, stride=1, kernel_size=5, act_cls=nn.ReLU, pool_first=True):
        super().__init__()

        nh1 = nf
        nh2 = nf
        nf_expanded = nf * expansion
        ni_expanded = ni * expansion

        self.convs = nn.Sequential(
            ConvLayer(ni_expanded, nh1, ks=1, stride=1, act_cls=act_cls),
            ConvLayer(nh1, nh2, ks=kernel_size, stride=stride, act_cls=act_cls),
            ConvLayer(nh2, nf_expanded, ks=1, stride=1, act_cls=None)
        )

        idpath = []

        if ni_expanded != nf_expanded:
            idpath.append(ConvLayer(ni_expanded, nf_expanded, ks=1, stride=1, act_cls=None))

        if stride != 1:
            pool = nn.AvgPool1d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

            if pool_first:
                idpath.insert(0, pool)
            else:
                idpath.append(pool)

        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.convs(x) + self.idpath(x))


def init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0.0)

    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)

    for child in m.children():
        init_cnn(child)


class XResNet1d(nn.Module):
    def __init__(
        self,
        expansion, layers, input_channels=12, num_classes=44, stem_szs=(32, 32, 64), kernel_size=5, kernel_size_stem=5,
        widen=1.0, act_cls=nn.ReLU, lin_ftrs_head=(128,), ps_head=0.5,
        bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True
    ):
        super().__init__()

        self.expansion = expansion
        self.act_cls = act_cls

        stem_channels = [input_channels, *stem_szs]

        self.stem = nn.Sequential(
            ConvLayer(stem_channels[0], stem_channels[1], ks=kernel_size_stem, stride=2, act_cls=act_cls),
            ConvLayer(stem_channels[1], stem_channels[2], ks=kernel_size_stem, stride=1, act_cls=act_cls),
            ConvLayer(stem_channels[2], stem_channels[3], ks=kernel_size_stem, stride=1, act_cls=act_cls)
        )

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs

        self.blocks = nn.ModuleList([
            self._make_layer(
                ni=block_szs[i], nf=block_szs[i + 1], blocks=n_blocks, stride=1 if i == 0 else 2, kernel_size=kernel_size
            )
            for i, n_blocks in enumerate(layers)
        ])

        self.head = create_head1d(
            block_szs[-1] * expansion,
            nc=num_classes,
            lin_ftrs=list(lin_ftrs_head) if lin_ftrs_head is not None else None,
            ps=ps_head,
            bn_final=bn_final_head,
            bn=bn_head,
            act=act_head,
            concat_pooling=concat_pooling
        )

        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size):
        return nn.Sequential(*[
            ResBlock(
                expansion=self.expansion,
                ni=ni if i == 0 else nf,
                nf=nf,
                stride=stride if i == 0 else 1,
                kernel_size=kernel_size,
                act_cls=self.act_cls
            )
            for i in range(blocks)
        ])

    def forward_features(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        for block in self.blocks:
            x = block(x)

        return x

    def forward_embedding(self, x):
        x = self.forward_features(x)
        x = self.head[0](x)  # AdaptiveConcatPool1d
        x = self.head[1](x)  # Flatten
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_output_layer(self):
        return self.head[-1]

    def set_output_layer(self, layer):
        self.head[-1] = layer


def xresnet1d101(num_classes=44, input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=(128,)):
    return XResNet1d(
        expansion=4,
        layers=[3, 4, 23, 3],
        num_classes=num_classes,
        input_channels=input_channels,
        kernel_size=kernel_size,
        ps_head=ps_head,
        lin_ftrs_head=lin_ftrs_head
    )


def encoder_state_dict_without_head(encoder):
    return {
        k: v
        for k, v in encoder.state_dict().items()
        if not k.startswith("head.")
    }


class ReconstructionAutoencoder(nn.Module):
    """
    reconstruction autoencoder

    used for:
        masking:    reconstruct masked temporal blocks
        denoising:  reconstruct clean ECG from noisy ECG

    input:  corrupted ECG crop: (B, 12, 250)
    output: reconstruction: (B, 12, 250)
    """

    def __init__(self, input_channels=12, input_size=250, kernel_size=5, ps_head=0.5, lin_ftrs_head=(128,)):
        super().__init__()

        self.input_size = input_size

        self.encoder = xresnet1d101(
            num_classes=1,
            input_channels=input_channels,
            kernel_size=kernel_size,
            ps_head=ps_head,
            lin_ftrs_head=lin_ftrs_head
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, input_channels, kernel_size=3, padding=1)
        )

        init_cnn(self.decoder)

    def forward(self, x_corrupted):
        features = self.encoder.forward_features(x_corrupted)

        upsampled = F.interpolate(features, size=self.input_size, mode="linear", align_corners=False)

        reconstruction = self.decoder(upsampled)
        return reconstruction

    def encoder_state_dict_without_head(self):
        return encoder_state_dict_without_head(self.encoder)


class ProjectionMLP(nn.Module):
    """
    projection head used by SimCLR and BYOL

    input: encoder embedding: (B, 512)
    output: projected embedding: (B, projection_dim)
    """

    def __init__(self, in_dim=512, hidden_dim=256, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

        init_cnn(self)

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    """
    BYOL prediction head

    input:  online projection: (B, projection_dim)
    output: prediction: (B, projection_dim)
    """

    def __init__(self, in_dim=128, hidden_dim=256, out_dim=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True)
        )

        init_cnn(self)

    def forward(self, x):
        return self.net(x)


class SimCLRModel(nn.Module):
    """
    SimCLR model
    uses:
        - xresnet1d101 encoder
        - projection MLP

    input:  ECG crop: (B, 12, 250)
    output: projection: (B, projection_dim)
    """

    def __init__(self, input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=(128,), projection_hidden_dim=256, projection_dim=128):
        super().__init__()

        self.encoder = xresnet1d101(
            num_classes=1,
            input_channels=input_channels,
            kernel_size=kernel_size,
            ps_head=ps_head,
            lin_ftrs_head=lin_ftrs_head
        )

        self.projector = ProjectionMLP(
            in_dim=512,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim
        )

    def forward(self, x):
        emb = self.encoder.forward_embedding(x)
        z = self.projector(emb)
        return z

    def encoder_state_dict_without_head(self):
        return encoder_state_dict_without_head(self.encoder)


class BYOLModel(nn.Module):
    """
    BYOL model

    components:
        - online_encoder
        - online_projector
        - online_predictor
        - target_encoder
        - target_projector

    the target network is updated by EMA outside the optimizer step
    """

    def __init__(
        self,
        input_channels=12, kernel_size=5, ps_head=0.5, lin_ftrs_head=(128,),
        projection_hidden_dim=256, projection_dim=128, prediction_hidden_dim=256
    ):
        super().__init__()

        self.online_encoder = xresnet1d101(
            num_classes=1,
            input_channels=input_channels,
            kernel_size=kernel_size,
            ps_head=ps_head,
            lin_ftrs_head=lin_ftrs_head
        )

        self.online_projector = ProjectionMLP(
            in_dim=512,
            hidden_dim=projection_hidden_dim,
            out_dim=projection_dim
        )

        self.online_predictor = PredictionMLP(
            in_dim=projection_dim,
            hidden_dim=prediction_hidden_dim,
            out_dim=projection_dim
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        self._set_target_requires_grad(False)

    def _set_target_requires_grad(self, requires_grad):
        for p in self.target_encoder.parameters():
            p.requires_grad = requires_grad

        for p in self.target_projector.parameters():
            p.requires_grad = requires_grad

    def online_forward(self, x):
        emb = self.online_encoder.forward_embedding(x)
        z = self.online_projector(emb)
        p = self.online_predictor(z)
        return p, z

    @torch.no_grad()
    def target_forward(self, x):
        emb = self.target_encoder.forward_embedding(x)
        z = self.target_projector(emb)
        return z

    def forward(self, x1, x2):
        p1, z1_online = self.online_forward(x1)
        p2, z2_online = self.online_forward(x2)

        with torch.no_grad():
            z1_target = self.target_forward(x1)
            z2_target = self.target_forward(x2)

        return {
            "p1": p1,
            "p2": p2,
            "z1_online": z1_online,
            "z2_online": z2_online,
            "z1_target": z1_target.detach(),
            "z2_target": z2_target.detach()
        }

    @torch.no_grad()
    def update_target_network(self, momentum=0.996):
        """
        EMA update:
            target = momentum * target + (1 - momentum) * online
        """

        online_modules = [self.online_encoder, self.online_projector]
        target_modules = [self.target_encoder, self.target_projector]

        for online_module, target_module in zip(online_modules, target_modules):
            for online_param, target_param in zip(online_module.parameters(), target_module.parameters()):
                target_param.data.mul_(momentum).add_(
                    online_param.data,
                    alpha=1.0 - momentum,
                )

            # update buffers too (BatchNorm running_mean/running_var)
            for online_buffer, target_buffer in zip(online_module.buffers(), target_module.buffers()):
                if torch.is_floating_point(target_buffer):
                    target_buffer.data.mul_(momentum).add_(online_buffer.data, alpha=1.0 - momentum)
                else:
                    target_buffer.data.copy_(online_buffer.data)

    def encoder_state_dict_without_head(self):
        return encoder_state_dict_without_head(self.online_encoder)

    def target_encoder_state_dict_without_head(self):
        return encoder_state_dict_without_head(self.target_encoder)
