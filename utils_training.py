import os, copy, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from compute import RunCompute, Stopwatch, count_params, forward_flops, hardware_info
from dataset import get_labels, get_split_indices, load_ptbxl_raw100
from drift import layer_activations, layerwise_cka, snapshot_encoder_params, weight_distance
from model import xresnet1d101
from utils import file_sha256, load_encoder_weights, safe_macro_auc, safe_macro_auprc, set_seed
from subsets import (
    STANDARD_FRACTIONS, frac_tag, label_statistics, nested_stratified_subsets, print_label_summary, save_label_statistics
)


"""
Shared downstream training for every method (scratch baseline and all SSL objectives), so every reported number comes from one code path.

Protocols (the classification head is always trained, at head_lr):
    - frozen:  encoder frozen, encoder in eval mode (BatchNorm statistics fixed)
    - partial: only the last residual stage (blocks[-1]) is trained, at encoder_lr; the rest frozen in eval mode
    - full:    the whole encoder is trained, at encoder_lr

Pairing: set_seed(seed) is called at the start of every run, before the model is built, so for a given seed every arm (scratch, frozen, partial, full; every objective)
starts from the same head initialization and sees the same minibatch order and crop offsets.
"""


PROTOCOLS = ("frozen", "partial", "full")


@dataclass
class FinetuneConfig:
    epochs: int = 50
    batch_size: int = 128
    head_lr: float = 1e-2
    weight_decay: float = 1e-2
    input_size: int = 250
    stride: int = 125
    eval_batch_size: int = 512
    kernel_size: int = 5
    ps_head: float = 0.5
    lin_ftrs_head: tuple = (128,)


# io helpers

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def atomic_json(path, obj):
    """write-then-rename, so a crash or a concurrent session never leaves a half-written file"""

    path = Path(path)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")

    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)

    os.replace(tmp, path)


def atomic_savez(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f".tmp{os.getpid()}_{path.name}")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def run_name(protocol, encoder_lr, head_lr):
    if protocol == "frozen":
        return f"frozen_head{head_lr:.0e}"

    return f"{protocol}_enc{encoder_lr:.0e}_head{head_lr:.0e}"


# data

def prepare_task(data_dir, task, subset_seed, fractions, n_strat_folds, task_dir):
    """
    loads PTB-XL, builds labels for `task`, the official split, and the stratified nested
    label-fraction subsets; writes label statistics and subset/label arrays to task_dir.
    """

    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    x, db = load_ptbxl_raw100(data_dir)
    y, label_names = get_labels(data_dir, db, task=task)
    train_idx, val_idx, test_idx = get_split_indices(db)

    ecg_ids = db.index.values
    patients = db["patient_id"].values

    all_fractions = sorted(set(STANDARD_FRACTIONS) | set(fractions))

    subsets, small_folds = nested_stratified_subsets(
        y[train_idx], patients[train_idx], all_fractions, n_folds=n_strat_folds, seed=subset_seed
    )

    per_label, summary = label_statistics(
        y[train_idx], patients[train_idx], subsets, y[val_idx], y[test_idx], label_names
    )

    print(f"task: {task} ({y.shape[1]} labels), subset seed: {subset_seed}")
    print_label_summary(summary)
    save_label_statistics(task_dir, per_label, summary)

    atomic_savez(
        task_dir / "subsets.npz",
        strat_small_fold=small_folds,
        train_ecg_id=ecg_ids[train_idx],
        **{f"idx_{frac_tag(f)}": subsets[f] for f in subsets},
        **{f"ecg_id_{frac_tag(f)}": ecg_ids[train_idx][subsets[f]] for f in subsets},
    )

    atomic_savez(
        task_dir / "labels.npz",
        label_names=np.array(label_names),
        y_val=y[val_idx],
        y_test=y[test_idx],
        val_ecg_id=ecg_ids[val_idx],
        test_ecg_id=ecg_ids[test_idx],
    )

    return {
        "x_train": x[train_idx],
        "y_train": y[train_idx],
        "x_val": x[val_idx],
        "y_val": y[val_idx],
        "x_test": x[test_idx],
        "y_test": y[test_idx],
        "patients_train": patients[train_idx],
        "subsets": subsets,
        "label_names": label_names,
        "num_classes": int(y.shape[1]),
    }


def make_chunks(x, input_size, stride):
    """
    x: (N, T, C) numpy -> (N, n_chunks, C, input_size) tensor
    identical windows to PTBXLCropDataset(chunkify=True, random_crop=False):
    starts at 0, stride, 2*stride, ... while start + input_size <= T (7 chunks for T=1000)
    """

    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).permute(0, 2, 1)
    return t.unfold(2, input_size, stride).permute(0, 2, 1, 3).contiguous()


def center_crops(x, input_size):
    """x: (N, T, C) numpy -> (N, C, input_size) tensor, same crop as random_crop=False"""

    start = (x.shape[1] - input_size) // 2
    crop = np.ascontiguousarray(x[:, start:start + input_size, :], dtype=np.float32)
    return torch.from_numpy(crop).permute(0, 2, 1).contiguous()


# model / protocol

def build_model(num_classes, cfg):
    return xresnet1d101(
        num_classes=num_classes,
        input_channels=12,
        kernel_size=cfg.kernel_size,
        ps_head=cfg.ps_head,
        lin_ftrs_head=cfg.lin_ftrs_head,
    )


def configure_protocol(model, protocol):
    """sets requires_grad and returns (trainable encoder params, head params)"""

    last_stage = f"blocks.{len(model.blocks) - 1}."
    encoder_params, head_params = [], []

    for name, p in model.named_parameters():
        if name.startswith("head."):
            p.requires_grad = True
            head_params.append(p)
            continue

        p.requires_grad = (protocol == "full") or (protocol == "partial" and name.startswith(last_stage))

        if p.requires_grad:
            encoder_params.append(p)

    return encoder_params, head_params


def set_train_mode(model, protocol):
    if protocol == "full":
        model.train()
        return

    model.eval()
    model.head.train()

    if protocol == "partial":
        model.blocks[-1].train()


def batch_bounds(n, batch_size):
    """
    same batches as a DataLoader with drop_last=False, except that a trailing batch of a single
    sample is merged into the previous batch (BatchNorm cannot train on one sample)
    """

    bounds = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]

    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] == 1:
        bounds[-2] = (bounds[-2][0], n)
        bounds.pop()

    return bounds


@torch.no_grad()
def predict_chunk_logits(model, chunks, batch_size):
    """chunks: (N, K, C, L) on device -> logits (N, K, n_classes) float32 on device"""

    model.eval()
    n, k = chunks.shape[:2]
    flat = chunks.reshape(n * k, *chunks.shape[2:])
    out = torch.cat([model(flat[i:i + batch_size]).float() for i in range(0, n * k, batch_size)])
    return out.reshape(n, k, -1)


def evaluate(model, chunks, y, batch_size):
    """record-level score = per-class max over chunk probabilities (the protocol of [3])"""

    logits = predict_chunk_logits(model, chunks, batch_size)
    probs = torch.sigmoid(logits).amax(dim=1).cpu().numpy()

    auroc, per_class_auroc = safe_macro_auc(y, probs)
    auprc, per_class_auprc = safe_macro_auprc(y, probs)

    return {
        "logits": logits.cpu().numpy(),
        "macro_auroc": auroc,
        "macro_auprc": auprc,
        "n_labels_evaluated": int(np.isfinite(per_class_auroc).sum()),
    }


# one run

def train_downstream(
    *, run_dir, method, task, protocol, encoder_lr, seed, subset_seed, fraction,
    pretrained_ckpt, data, cfg, device, provenance, init_state_path=None,
):
    run_dir = Path(run_dir)

    if (run_dir / "metrics.json").exists():
        print(f"skip, already done: {run_dir}")
        with open(run_dir / "metrics.json") as f:
            return json.load(f)

    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got {protocol!r}")

    if protocol == "frozen" and encoder_lr != 0.0:
        raise ValueError("frozen protocol requires encoder_lr == 0")

    if pretrained_ckpt is None and protocol != "full":
        raise ValueError("a randomly initialized encoder is only trained with the full protocol")

    run_dir.mkdir(parents=True, exist_ok=True)

    subset_idx = data["subsets"][fraction]
    x_train = data["x_train"][subset_idx]
    y_train = data["y_train"][subset_idx]
    np.save(run_dir / "train_subset_indices.npy", subset_idx)

    # model, seeded per run so all arms with this seed are paired
    set_seed(seed)
    model = build_model(data["num_classes"], cfg)

    if pretrained_ckpt is not None:
        model = load_encoder_weights(model, pretrained_ckpt)
    elif init_state_path is not None and not Path(init_state_path).exists():
        torch.save(model.state_dict(), init_state_path)  # reference point for scratch drift

    model = model.to(device)

    encoder_params, head_params = configure_protocol(model, protocol)

    param_groups = []

    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr, "name": "encoder"})

    param_groups.append({"params": head_params, "lr": cfg.head_lr, "name": "head"})

    optimizer = torch.optim.AdamW(param_groups, lr=cfg.head_lr, weight_decay=cfg.weight_decay)

    bounds = batch_bounds(len(x_train), cfg.batch_size)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in param_groups],
        epochs=cfg.epochs,
        steps_per_epoch=len(bounds),
    )

    criterion = nn.BCEWithLogitsLoss()

    # data on device; training crops are drawn on the GPU (one random crop per record per epoch)
    x_tr = torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)).to(device)
    y_tr = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32)).to(device)
    n_time = x_tr.shape[1]
    window = torch.arange(cfg.input_size, device=device)

    val_chunks, test_chunks, probe = data["val_chunks"], data["test_chunks"], data["probe"]

    # drift reference: the encoder as it starts this run (pretrained, or random init)
    model.eval()
    ref_acts = layer_activations(model, probe, cfg.eval_batch_size)
    ref_params = snapshot_encoder_params(model)

    n_trainable_encoder = sum(p.numel() for p in encoder_params)
    n_trainable_head = sum(p.numel() for p in head_params)

    best_val_auroc, best_state, best_epoch = -np.inf, None, None
    history = []
    n_steps = 0

    rc = RunCompute().start()

    for epoch in range(1, cfg.epochs + 1):
        set_train_mode(model, protocol)

        sw = Stopwatch().start()
        perm = torch.randperm(len(x_tr), device=device)
        loss_sum = torch.zeros((), device=device)

        for start, end in bounds:
            idx = perm[start:end]
            offsets = torch.randint(0, n_time - cfg.input_size + 1, (len(idx),), device=device)
            xb = x_tr[idx[:, None], offsets[:, None] + window[None, :]].permute(0, 2, 1).contiguous()
            yb = y_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.detach()
            n_steps += 1

        train_loss = float(loss_sum) / len(bounds)
        train_s = sw.stop()

        sw = Stopwatch().start()
        val = evaluate(model, val_chunks, data["y_val"], cfg.eval_batch_size)
        drift = layerwise_cka(ref_acts, layer_activations(model, probe, cfg.eval_batch_size))
        drift.update(weight_distance(model, ref_params))
        eval_s = sw.stop()

        rc.add("train_s", train_s)
        rc.add("eval_s", eval_s)

        if val["macro_auroc"] > best_val_auroc:
            best_val_auroc = val["macro_auroc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        lrs = dict(zip([g["name"] for g in param_groups], scheduler.get_last_lr()))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_auroc": val["macro_auroc"],
            "val_macro_auprc": val["macro_auprc"],
            "best_val_macro_auroc": best_val_auroc,
            "lr_encoder": lrs.get("encoder", 0.0),
            "lr_head": lrs["head"],
            "train_time_s": train_s,
            "eval_time_s": eval_s,
            **drift,
        }

        history.append(row)

        print(
            f"[{method} {protocol} enc={encoder_lr:.0e} frac={fraction:.2f} seed={seed}] "
            f"epoch {epoch}/{cfg.epochs} loss={train_loss:.4f} "
            f"val_auroc={val['macro_auroc']:.4f} best={best_val_auroc:.4f} "
            f"cka_stage4={drift['cka_stage4']:.3f} wdist={drift['wdist_encoder']:.3f} "
            f"({train_s:.1f}s + {eval_s:.1f}s)"
        )

    # best checkpoint, final evaluation
    model.load_state_dict(best_state)

    val = evaluate(model, val_chunks, data["y_val"], cfg.eval_batch_size)
    test = evaluate(model, test_chunks, data["y_test"], cfg.eval_batch_size)

    compute = rc.finish()
    compute.update({
        "n_optimizer_steps": n_steps,
        "n_train_samples_seen": int(cfg.epochs * len(x_train)),
        "train_samples_per_s": cfg.epochs * len(x_train) / max(compute.get("train_s", 0.0), 1e-9),
        "n_trainable_params_encoder": int(n_trainable_encoder),
        "n_trainable_params_head": int(n_trainable_head),
    })

    np.save(run_dir / "val_logits.npy", val["logits"].astype(np.float32))
    np.save(run_dir / "test_logits.npy", test["logits"].astype(np.float32))

    torch.save(
        {"model": {k: v.cpu() for k, v in model.state_dict().items()}, "best_epoch": best_epoch},
        run_dir / "best_model.pt",
    )

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    run_config = {
        "method": method,
        "task": task,
        "protocol": protocol,
        "encoder_lr": encoder_lr,
        "seed": seed,
        "subset_seed": subset_seed,
        "fraction": fraction,
        "finetune_config": asdict(cfg),
        **provenance,
    }

    atomic_json(run_dir / "run_config.json", run_config)
    atomic_json(run_dir / "compute.json", compute)

    metrics = {
        "method": method,
        "task": task,
        "protocol": protocol,
        "encoder_lr": encoder_lr,
        "head_lr": cfg.head_lr,
        "seed": seed,
        "subset_seed": subset_seed,
        "fraction": fraction,
        "n_train_records": int(len(x_train)),
        "n_train_patients": int(len(np.unique(data["patients_train"][subset_idx]))),
        "n_optimizer_steps": n_steps,
        "best_epoch": best_epoch,
        "final_train_loss": history[-1]["train_loss"],
        "best_val_macro_auroc": float(best_val_auroc),
        "val_macro_auroc": val["macro_auroc"],
        "val_macro_auprc": val["macro_auprc"],
        "test_macro_auroc": test["macro_auroc"],
        "test_macro_auprc": test["macro_auprc"],
        "n_test_labels_evaluated": test["n_labels_evaluated"],
        "pooling": "max_prob_over_chunks",
        "pretrained_ckpt_sha256": provenance.get("pretrained_ckpt_sha256")
    }

    atomic_json(run_dir / "metrics.json", metrics)  # written last: its presence marks the run as complete

    print(
        f"done: {run_dir} | best epoch {best_epoch} | "
        f"test macro AUROC {test['macro_auroc']:.4f}, AUPRC {test['macro_auprc']:.4f} | "
        f"{compute['wall_clock_s'] / 60:.1f} min, peak mem {compute.get('peak_memory_allocated_mb', 0):.0f} MB, "
        f"energy {compute['energy_j'] if compute['energy_j'] is not None else float('nan'):.0f} J"
    )

    del model, optimizer, x_tr, y_tr, best_state

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# the grid

def collect_metrics(task_dir):
    rows = []

    for path in sorted(Path(task_dir).glob("*/*pct/metrics.json")):
        with open(path) as f:
            rows.append(json.load(f))

    return rows


def run_downstream_grid(
    *, method, task, seed, subset_seed, pretrained_ckpt, protocols, encoder_lrs, fractions,
    data, task_dir, cfg, device, extra_config=None,
):
    """
    loops label fraction (smallest first) x protocol x encoder LR; every finished run is skipped
    on restart, so a Kaggle session can be resumed or the grid split across sessions.
    """

    task_dir = Path(task_dir)

    for p in protocols:
        if p not in PROTOCOLS:
            raise ValueError(f"unknown protocol {p!r}")

        if pretrained_ckpt is None and p != "full":
            raise ValueError("the scratch baseline only supports the 'full' protocol")

    for f in fractions:
        if f not in data["subsets"]:
            raise ValueError(f"fraction {f} has no subset")

    provenance = {
        "hardware": hardware_info(),
        "pretrained_ckpt": str(pretrained_ckpt) if pretrained_ckpt is not None else None,
        "pretrained_ckpt_sha256": file_sha256(pretrained_ckpt) if pretrained_ckpt is not None else None,
    }

    # model size and FLOPs of the downstream network (encoder + head), once per task dir
    stats_path = task_dir / "downstream_model_stats.json"

    if not stats_path.exists():
        m = build_model(data["num_classes"], cfg)
        atomic_json(stats_path, {
            "params_total": count_params(m),
            "params_encoder": count_params(m) - count_params(m.head),
            "params_head": count_params(m.head),
            "params_last_stage": count_params(m.blocks[-1]),
            "forward_flops_per_crop": forward_flops(m, torch.zeros(2, 12, cfg.input_size)),
            "flops_note": "torch FlopCounterMode, FLOPs = 2 x MACs, one (12, input_size) crop",
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_json(task_dir / f"invocation_{stamp}_{os.getpid()}.json", {
        "method": method,
        "task": task,
        "seed": seed,
        "subset_seed": subset_seed,
        "protocols": list(protocols),
        "encoder_lrs": list(encoder_lrs),
        "fractions": list(fractions),
        "finetune_config": asdict(cfg),
        **(extra_config or {}),
        **provenance,
    })

    # evaluation tensors live on the device for the whole grid
    data = dict(data)
    data["val_chunks"] = make_chunks(data["x_val"], cfg.input_size, cfg.stride).to(device)
    data["test_chunks"] = make_chunks(data["x_test"], cfg.input_size, cfg.stride).to(device)
    data["probe"] = center_crops(data["x_val"], cfg.input_size).to(device)

    init_state_path = task_dir / f"init_state_seed{seed}.pt" if pretrained_ckpt is None else None

    results = []

    for frac in sorted(fractions):
        for protocol in protocols:
            for enc_lr in ([0.0] if protocol == "frozen" else encoder_lrs):
                run_dir = task_dir / run_name(protocol, enc_lr, cfg.head_lr) / frac_tag(frac)

                results.append(train_downstream(
                    run_dir=run_dir,
                    method=method,
                    task=task,
                    protocol=protocol,
                    encoder_lr=float(enc_lr),
                    seed=seed,
                    subset_seed=subset_seed,
                    fraction=frac,
                    pretrained_ckpt=pretrained_ckpt,
                    data=data,
                    cfg=cfg,
                    device=device,
                    provenance=provenance,
                    init_state_path=init_state_path,
                ))

    atomic_json(task_dir / "all_metrics.json", collect_metrics(task_dir))

    return results
