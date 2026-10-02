import numpy as np
import torch
from compute import count_params


"""
Helpers shared by the SSL pretraining scripts.

PRETRAIN mode (environment variable, default "auto"):
    - auto:  reuse <method>_seed<seed>/pretrain/pretrain_last.pt if it exists and is complete, else pretrain
    - reuse: the checkpoint must exist and be complete, otherwise stop (use this when re-running downstream experiments on encoders that were already pretrained, so nothing is ever silently re-pretrained)
    - train: the checkpoint must not exist, otherwise stop (never overwrites an encoder)

PROFILE_PRETRAIN_EPOCHS=k (environment variable): run k pretraining epochs with the full-length schedule,
log time / memory / energy to <method>_seed<seed>/pretrain_profile/, save no checkpoint, then exit.
"""


def resolve_pretrain_mode(mode, ckpt_path, expected_epochs):
    if mode not in ("auto", "reuse", "train"):
        raise ValueError(f"PRETRAIN must be auto, reuse or train, got {mode!r}")

    exists = ckpt_path.exists()

    if exists:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        epoch = int(ckpt.get("epoch", -1))
        del ckpt

        if epoch != expected_epochs:
            raise RuntimeError(
                f"{ckpt_path} stopped at epoch {epoch} of {expected_epochs}: incomplete pretraining. "
                f"delete it to pretrain again, or restore the complete checkpoint."
            )

    if mode == "reuse" and not exists:
        raise FileNotFoundError(
            f"PRETRAIN=reuse but {ckpt_path} does not exist. copy the original pretrain/ folder there first."
        )

    if mode == "train" and exists:
        raise FileExistsError(f"PRETRAIN=train but {ckpt_path} already exists; refusing to overwrite it.")

    return "reuse" if exists else "train"


def pretraining_param_counts(model):
    """
    parameters actually used during pretraining. the xresnet1d101 classification head inside each
    SSL encoder is never called during pretraining and is excluded.
    """

    used = [(n, p) for n, p in model.named_parameters() if "head" not in n.split(".")]

    return {
        "params_used_total": int(sum(p.numel() for _, p in used)),
        "params_used_trainable": int(sum(p.numel() for _, p in used if p.requires_grad)),
        "params_encoder": int(sum(p.numel() for n, p in used if n.split(".")[0] in ("encoder", "online_encoder"))),
        "params_all_including_unused_heads": count_params(model),
    }


def summarize_pretrain_compute(compute, epoch_times, schedule_epochs, steps_per_epoch, extra):
    epoch_times = np.asarray(epoch_times, dtype=np.float64)
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times  # epoch 1 includes CUDA warm-up

    out = dict(compute)
    out.update({
        "n_epochs_run": int(len(epoch_times)),
        "schedule_epochs": int(schedule_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "epoch_time_s_mean": float(epoch_times.mean()),
        "epoch_time_s_median_excl_first": float(np.median(steady)),
        "projected_full_pretraining_h": float(np.median(steady) * schedule_epochs / 3600.0),
    })

    if compute.get("energy_j") is not None and len(epoch_times) > 0:
        per_epoch_j = compute["energy_j"] / len(epoch_times)
        out["energy_per_epoch_j"] = per_epoch_j
        out["projected_full_pretraining_kwh"] = per_epoch_j * schedule_epochs / 3.6e6

    out.update(extra)
    return out
