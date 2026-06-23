import torch
import torch.nn.functional as F


def reconstruction_loss(recon, clean, loss_mask):
    """
    recon:      (B, 12, 250)
    clean:      (B, 12, 250)
    loss_mask:  (B, 1, 250), where 1 means compute loss here
    """

    loss_mask = loss_mask.expand_as(clean)

    loss = F.smooth_l1_loss(recon * loss_mask, clean * loss_mask, reduction="sum")

    denom = loss_mask.sum().clamp_min(1.0)
    return loss / denom


def nt_xent_loss(z1, z2, temperature=0.2):
    """
    SimCLR NT-Xent loss

    z1: (B, D)
    z2: (B, D)

    returns:  scalar loss
    """

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("z1 and z2 must both be 2D tensors")

    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}")

    batch_size = z1.size(0)

    if batch_size < 2:
        raise ValueError("NT-Xent loss requires batch_size >= 2")

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    logits = torch.matmul(z, z.T) / temperature  # (2B, 2B)

    # mask self-similarity
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, -1e9)

    # positive for i is i+B if i<B, else i-B
    labels = torch.arange(2 * batch_size, device=z.device)
    labels = (labels + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(logits, labels)
    return loss


def byol_loss(p, z):
    """
    BYOL negative cosine similarity loss

    p: online prediction
    z: target projection, detached before/inside loss

    returns: scalar loss
    """

    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)

    return 2.0 - 2.0 * (p * z).sum(dim=1).mean()


def byol_symmetric_loss(outputs):
    """
    symmetric BYOL loss
        - p1 predicts target z2
        - p2 predicts target z1
    """

    loss_12 = byol_loss(outputs["p1"], outputs["z2_target"])
    loss_21 = byol_loss(outputs["p2"], outputs["z1_target"])

    return 0.5 * (loss_12 + loss_21)
