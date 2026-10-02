import torch


"""
Representation drift between the starting encoder (pretrained, or random init for scratch) and the encoder during/after downstream training.

    - Layer-wise linear CKA (Kornblith et al., 2019) on a fixed probe set, both encoders in eval mode, so changes in BatchNorm running statistics count as drift
    - Relative L2 weight distance per stage, ||w - w0|| / ||w0||, over parameters only (no BN buffers)

Layers: stem (after max pool), stage1..stage4 (time-averaged), embedding (concat max+avg pool, 512-d)
"""


LAYERS = ["stem", "stage1", "stage2", "stage3", "stage4", "embedding"]
GROUPS = ["stem", "stage1", "stage2", "stage3", "stage4"]


def _group_of(param_name):
    if param_name.startswith("stem."):
        return "stem"

    if param_name.startswith("blocks."):
        return f"stage{int(param_name.split('.')[1]) + 1}"

    return None  # head, excluded


@torch.no_grad()
def layer_activations(model, x, batch_size=512):
    """
    model: XResNet1d (caller puts it in eval mode)
    x:     (N, 12, L) tensor on the model's device
    returns {layer: (N, D) float64 tensor}
    """

    feats = {k: [] for k in LAYERS}

    for i in range(0, len(x), batch_size):
        h = model.maxpool(model.stem(x[i:i + batch_size]))
        feats["stem"].append(h.mean(dim=-1))

        for j, block in enumerate(model.blocks):
            h = block(h)
            feats[f"stage{j + 1}"].append(h.mean(dim=-1))

        emb = model.head[1](model.head[0](h))  # AdaptiveConcatPool1d -> Flatten
        feats["embedding"].append(emb)

    return {k: torch.cat(v).double() for k, v in feats.items()}


def linear_cka(x, y, eps=1e-12):
    """x: (N, D1), y: (N, D2), same rows. returns a float in [0, 1]"""

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    hsic = (y.T @ x).pow(2).sum()
    norm_x = (x.T @ x).pow(2).sum().sqrt()
    norm_y = (y.T @ y).pow(2).sum().sqrt()

    return float(hsic / (norm_x * norm_y + eps))


def layerwise_cka(ref_acts, acts):
    return {f"cka_{k}": linear_cka(ref_acts[k], acts[k]) for k in LAYERS}


@torch.no_grad()
def snapshot_encoder_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if _group_of(n) is not None}


@torch.no_grad()
def weight_distance(model, ref_params):
    num = {g: 0.0 for g in GROUPS}
    den = {g: 0.0 for g in GROUPS}

    for n, p in model.named_parameters():
        g = _group_of(n)

        if g is None:
            continue

        p0 = ref_params[n]
        num[g] += float((p.detach() - p0).double().pow(2).sum())
        den[g] += float(p0.double().pow(2).sum())

    out = {f"wdist_{g}": (num[g] ** 0.5) / (den[g] ** 0.5 + 1e-12) for g in GROUPS}
    out["wdist_encoder"] = (sum(num.values()) ** 0.5) / (sum(den.values()) ** 0.5 + 1e-12)

    return out
