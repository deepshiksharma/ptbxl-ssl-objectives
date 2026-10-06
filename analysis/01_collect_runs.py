# Step 1: Collect every downstream run into one table, and audit the run tree

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import (
    FRACTIONS, N_FINETUNE_EPOCHS, N_PRETRAIN_EPOCHS, SSL_METHODS,
    array_digest, expected_configs, file_sha256, find_sessions, frac_tag, read_json, run_name, write_text
)

"""
python 01_collect_runs.py <runs-dir> <out-dir>

writes:
    <out-dir>/runs.csv    one row per downstream run (metrics, compute, drift at the selected checkpoint)
    <out-dir>/audit.txt   integrity report (also printed)
"""


USAGE = """Usage:
    python 01_collect_runs.py <runs-dir> <out-dir>
        runs-dir: folder containing the <method>_seed<seed>/ session folders
        out-dir:  folder for analysis outputs (created if missing)
"""

if len(sys.argv) != 3:
    raise ValueError(USAGE)

RUNS_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)


TASK = "diagnostic"

METRIC_KEYS = [
    "protocol", "encoder_lr", "head_lr", "fraction", "subset_seed", "n_train_records", "n_train_patients",
    "n_optimizer_steps", "best_epoch", "final_train_loss", "best_val_macro_auroc", "val_macro_auroc",
    "val_macro_auprc", "test_macro_auroc", "test_macro_auprc", "n_test_labels_evaluated", "pretrained_ckpt_sha256",
]

COMPUTE_KEYS = [
    "wall_clock_s", "train_s", "eval_s", "peak_memory_allocated_mb", "energy_j", "energy_method", "mean_power_w",
    "train_samples_per_s", "n_trainable_params_encoder", "n_trainable_params_head",
]


def collect_run(method, seed, run_dir):
    m = read_json(run_dir / "metrics.json")
    row = {"method": method, "seed": seed, "config": run_dir.parent.name, "run_dir": str(run_dir.relative_to(RUNS_DIR))}
    row.update({k: m.get(k) for k in METRIC_KEYS})

    if (run_dir / "compute.json").exists():
        c = read_json(run_dir / "compute.json")
        row.update({k: c.get(k) for k in COMPUTE_KEYS})

    if (run_dir / "run_config.json").exists():
        row["gpu_name"] = read_json(run_dir / "run_config.json").get("hardware", {}).get("gpu_name")

    # drift of the checkpoint that was evaluated (the best-validation epoch)
    hist = pd.read_csv(run_dir / "history.csv")
    row["n_epochs_logged"] = len(hist)
    best = hist[hist["epoch"] == m["best_epoch"]]

    if len(best):
        row.update({k: best.iloc[0][k] for k in hist.columns if k.startswith(("cka_", "wdist_"))})

    row["has_logits"] = (run_dir / "val_logits.npy").exists() and (run_dir / "test_logits.npy").exists()
    row["has_checkpoint"] = (run_dir / "best_model.pt").exists()

    return row


def npz_digest(path, keys_prefix=None):
    with np.load(path, allow_pickle=False) as z:
        return array_digest({k: z[k] for k in z.files if keys_prefix is None or k.startswith(keys_prefix)})


sessions = find_sessions(RUNS_DIR)

if not sessions:
    raise FileNotFoundError(f"no <method>_seed<seed> folders in {RUNS_DIR}")

rows = []
report = [f"runs dir: {RUNS_DIR}", f"task: {TASK}", ""]
problems = 0


def flag(msg, examples=()):
    # one report line per problem; long lists are shortened to a count and a few examples
    global problems
    problems += 1
    examples = list(examples)
    more = f" (+{len(examples) - 5} more)" if len(examples) > 5 else ""
    report.append("  PROBLEM: " + msg + (": " + ", ".join(examples[:5]) + more if examples else ""))


# per session: expected runs present, histories complete, pretraining complete
report.append("sessions:")
labels_digest, subsets_digest, label_counts = {}, {}, {}

for method, seed, path in sessions:
    task_dir = path / TASK

    if not task_dir.exists():
        flag(f"{path.name}: no {TASK}/ folder")
        continue

    n_found, missing = 0, []

    for protocol, lr in expected_configs(method):
        for frac in FRACTIONS:
            run_dir = task_dir / run_name(protocol, lr) / frac_tag(frac)

            if not (run_dir / "metrics.json").exists():
                missing.append(f"{run_name(protocol, lr)}/{frac_tag(frac)}")
                continue

            rows.append(collect_run(method, seed, run_dir))
            n_found += 1

    if missing:
        flag(f"{path.name}: {len(missing)} runs missing", missing)

    n_expected = len(expected_configs(method)) * len(FRACTIONS)
    line = f"  {path.name:18s} {n_found}/{n_expected} runs"

    if method in SSL_METHODS:
        pre = path / "pretrain"
        hist = pre / "pretrain_history.csv"
        n_pre = len(pd.read_csv(hist)) if hist.exists() else 0
        line += f" | pretraining epochs {n_pre}/{N_PRETRAIN_EPOCHS}"

        if n_pre != N_PRETRAIN_EPOCHS:
            flag(f"{path.name}: pretraining history has {n_pre} epochs")

        if not (pre / "pretrain_compute.json").exists():
            flag(f"{path.name}: pretrain/pretrain_compute.json missing")

        ckpt = pre / "pretrain_last.pt"
        used = {r["pretrained_ckpt_sha256"] for r in rows if r["method"] == method and r["seed"] == seed}

        if len(used) != 1:
            flag(f"{path.name}: {len(used)} different encoders used across its runs")
        elif ckpt.exists() and file_sha256(ckpt) not in used:
            flag(f"{path.name}: pretrain_last.pt is not the encoder its runs used")
        elif not ckpt.exists():
            flag(f"{path.name}: pretrain/pretrain_last.pt missing")
        else:
            line += " | encoder checksum matches"

    report.append(line)

    labels_digest[path.name] = npz_digest(task_dir / "labels.npz")
    subsets_digest[(method, seed)] = npz_digest(task_dir / "subsets.npz", keys_prefix="idx_")
    label_counts[(method, seed)] = read_json(task_dir / "label_summary.json")

runs = pd.DataFrame(rows)
runs.to_csv(OUT_DIR / "runs.csv", index=False)


# consistency across sessions
report += ["", "consistency:"]

if len(set(labels_digest.values())) == 1:
    report.append(f"  validation/test labels identical across all {len(labels_digest)} sessions")
else:
    flag(f"validation/test labels differ between sessions: {labels_digest}")

for seed in sorted({s for _, s in subsets_digest}):
    digests = {m: d for (m, s), d in subsets_digest.items() if s == seed}

    if len(set(digests.values())) == 1:
        sizes = {k: v["n_records"] for k, v in label_counts[(next(iter(digests)), seed)].items() if k.endswith("pct")}
        report.append(f"  seed {seed}: identical labeled subsets across {len(digests)} methods, sizes {sizes}")
    else:
        flag(f"seed {seed}: labeled subsets differ between methods")

if (runs["subset_seed"] != runs["seed"]).any():
    flag("some runs used a subset seed different from their training seed")

bad_hist = runs[runs["n_epochs_logged"] != N_FINETUNE_EPOCHS]

if len(bad_hist):
    flag(f"{len(bad_hist)} runs do not have {N_FINETUNE_EPOCHS} epochs in history.csv", bad_hist["run_dir"])

n_labels = sorted(int(n) for n in runs["n_test_labels_evaluated"].unique())
report.append(f"  labels evaluated on test: {n_labels}")

for col, what in (("has_logits", "val/test logits"), ("has_checkpoint", "best_model.pt")):
    missing = runs[~runs[col].astype(bool)]

    if len(missing):
        flag(f"{len(missing)} runs without {what}", missing["run_dir"])

gpus = runs.groupby(["method", "seed"])["gpu_name"].agg(lambda s: "/".join(sorted(set(map(str, s)))))
report.append("  GPUs: " + ", ".join(sorted(set(gpus.values))))

energy = runs["energy_method"].dropna().unique()
report.append(f"  energy measured by: {', '.join(map(str, energy)) if len(energy) else 'NONE (nvidia-ml-py missing?)'}")

# summary
seeds = sorted(int(s) for s in runs["seed"].unique())
report += ["", f"collected {len(runs)} runs from {len(sessions)} sessions, seeds {seeds}"]
report.append(f"{problems} problem(s) found" if problems else "no problems found")
report.append(f"wrote {OUT_DIR / 'runs.csv'}")

write_text(OUT_DIR / "audit.txt", report)
print("\n".join(report))
