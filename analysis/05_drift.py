# Step 5: Representation drift of the pretrained encoders during partial and full fine-tuning

from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import get_split_indices, load_ptbxl_raw100
from drift import LAYERS, layer_activations, linear_cka
from utils import load_encoder_weights
from utils_training import FinetuneConfig, build_model, center_crops
from common import (
    ENCODER_LRS, FRACTIONS, METHOD_NAMES, SSL_METHODS, frac_tag, load_runs, write_text
)

"""
python 05_drift.py <data-dir> <runs-dir> <out-dir>

the CKA logged during training compares encoders in eval mode with each encoder's own BatchNorm statistics.
the pretrained encoder's statistics were estimated on corrupted/augmented pretraining inputs, so they shift
towards clean data during fine-tuning even when the weights barely move. here, the BatchNorm statistics of
both encoders are re-estimated on the same clean training crops before comparing them, so the CKA reflects
changes in the weights only. both versions are reported.

for every partial and full run of every SSL objective (the selected checkpoint):
    layer-wise linear CKA (stem, stage1-4, embedding) to the pretrained encoder, after BatchNorm recalibration
reference values (same probe set, same recalibration):
    pretrained vs randomly initialized encoder, and pretrained vs pretrained across seeds (same objective)

reads:
    <out-dir>/runs.csv                                         (from 01_collect_runs.py)
    <runs-dir>/<method>_seed<seed>/pretrain/pretrain_last.pt, .../best_model.pt
    <runs-dir>/baseline_seed<seed>/<task>/init_state_seed<seed>.pt
writes:
    <out-dir>/drift_runs.csv, drift_reference.csv, drift.txt, fig_drift.pdf/.png
"""


USAGE = """Usage:
    python 05_drift.py <data-dir> <runs-dir> <out-dir>
        data-dir: PTB-XL 100 Hz directory (contains ptbxl_database.csv; raw100.npy cache is reused if present)
        runs-dir: folder containing the <method>_seed<seed>/ session folders
        out-dir:  folder with runs.csv from 01_collect_runs.py; outputs are written here
"""

if len(sys.argv) != 4:
    raise ValueError(USAGE)

DATA_DIR, RUNS_DIR, OUT_DIR = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
TASK = "diagnostic"

N_CALIB = 1024  # clean training crops used to re-estimate BatchNorm statistics
CALIB_SEED = 0
BATCH_SIZE = 256
PROTOCOLS = ["partial", "full"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CFG = FinetuneConfig()


@torch.no_grad()
def recalibrated_activations(model, calib, probe):
    """re-estimate every BatchNorm's statistics on `calib` (cumulative average), then activations on `probe`"""

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = None

    model.train()

    for i in range(0, len(calib), BATCH_SIZE):
        model(calib[i:i + BATCH_SIZE])

    model.eval()

    return layer_activations(model, probe, BATCH_SIZE)


def load_finetuned(run_dir, num_classes):
    model = build_model(num_classes, CFG)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)["model"])
    return model.to(DEVICE)


def load_pretrained(method, seed, num_classes):
    model = build_model(num_classes, CFG)
    return load_encoder_weights(model, RUNS_DIR / f"{method}_seed{seed}" / "pretrain" / "pretrain_last.pt").to(DEVICE)


def pct(frac):
    return f"{int(round(frac * 100))}%"


# data: clean training crops for recalibration, validation center crops as the probe set (as in training)
x, db = load_ptbxl_raw100(DATA_DIR, cache_dir="../data/cache")
train_idx, val_idx, _ = get_split_indices(db)

rng = np.random.default_rng(CALIB_SEED)
calib_idx = rng.choice(train_idx, size=min(N_CALIB, len(train_idx)), replace=False)
offsets = rng.integers(0, x.shape[1] - CFG.input_size + 1, size=len(calib_idx))
calib = np.stack([x[i, o:o + CFG.input_size].T for i, o in zip(calib_idx, offsets)]).astype(np.float32)
calib = torch.from_numpy(calib).to(DEVICE)
probe = center_crops(x[val_idx], CFG.input_size).to(DEVICE)
del x

runs = load_runs(OUT_DIR)
seeds = sorted(int(s) for s in runs["seed"].unique())

with np.load(RUNS_DIR / f"baseline_seed{seeds[0]}" / TASK / "labels.npz", allow_pickle=False) as z:
    num_classes = len(z["label_names"])
todo = runs[runs["method"].isin(SSL_METHODS) & runs["protocol"].isin(PROTOCOLS)]

print(f"device {DEVICE}, {len(todo)} runs, {len(calib)} calibration crops, {len(probe)} probe crops")


# pretrained (reference) activations, once per session
pretrained_acts = {}

for m in SSL_METHODS:
    for s in seeds:
        pretrained_acts[(m, s)] = recalibrated_activations(load_pretrained(m, s, num_classes), calib, probe)


# every partial / full run vs its own pretrained encoder
rows = []

for i, r in enumerate(todo.itertuples()):
    run_dir = RUNS_DIR / f"{r.method}_seed{r.seed}" / TASK / r.config / frac_tag(r.fraction)
    acts = recalibrated_activations(load_finetuned(run_dir, num_classes), calib, probe)
    ref = pretrained_acts[(r.method, r.seed)]

    rows.append({
        "method": r.method, "seed": r.seed, "protocol": r.protocol, "encoder_lr": r.encoder_lr, "fraction": r.fraction,
        **{f"cka_bn_{k}": linear_cka(ref[k], acts[k]) for k in LAYERS},
    })

    if (i + 1) % 30 == 0:
        print(f"  {i + 1}/{len(todo)}")

logged_cols = [c for c in runs.columns if c.startswith(("cka_", "wdist_"))]
drift = pd.DataFrame(rows).merge(
    runs[["method", "seed", "protocol", "encoder_lr", "fraction", "test_macro_auroc"] + logged_cols],
    on=["method", "seed", "protocol", "encoder_lr", "fraction"],
)
drift.to_csv(OUT_DIR / "drift_runs.csv", index=False)


# reference values
ref_rows = []

for m in SSL_METHODS:
    for s in seeds:
        init_path = RUNS_DIR / f"baseline_seed{s}" / TASK / f"init_state_seed{s}.pt"

        if init_path.exists():
            random_model = build_model(num_classes, CFG)
            random_model.load_state_dict(torch.load(init_path, map_location="cpu", weights_only=True))
            acts = recalibrated_activations(random_model.to(DEVICE), calib, probe)
            ref_rows.append({"comparison": "pretrained_vs_random_init", "method": m, "seeds": f"{s}",
                             **{f"cka_bn_{k}": linear_cka(pretrained_acts[(m, s)][k], acts[k]) for k in LAYERS}})

    for a, b in combinations(seeds, 2):
        ref_rows.append({"comparison": "pretrained_vs_pretrained_other_seed", "method": m, "seeds": f"{a}-{b}",
                         **{f"cka_bn_{k}": linear_cka(pretrained_acts[(m, a)][k], pretrained_acts[(m, b)][k]) for k in LAYERS}})

reference = pd.DataFrame(ref_rows)
reference.to_csv(OUT_DIR / "drift_reference.csv", index=False)


# text report: means over objectives and seeds
def table(col, title, fmt="{:.3f}"):
    means = drift.groupby(["protocol", "encoder_lr", "fraction"])[col].mean()
    lines = [title, f"{'':22s}" + "".join(f"{pct(f):>8s}" for f in FRACTIONS)]

    for p in PROTOCOLS:
        for lr in ENCODER_LRS:
            label = f"{p} enc {lr:.0e}"
            lines.append(f"{label:22s}" + "".join(f"{fmt.format(means.loc[(p, lr, f)]):>8s}" for f in FRACTIONS))

    return lines + [""]


report = [f"seeds {seeds}, objectives {SSL_METHODS}; values are means over objectives and seeds", ""]
report += table("cka_bn_embedding", "CKA to the pretrained encoder, embedding, BatchNorm recalibrated (weights only)")
report += table("cka_embedding", "CKA to the pretrained encoder, embedding, as logged (weights + BatchNorm statistics)")
report += table("cka_bn_stage4", "CKA to the pretrained encoder, stage 4, BatchNorm recalibrated")
report += table("wdist_encoder", "relative weight distance ||w - w0|| / ||w0||, whole encoder")

report.append("reference CKA (embedding, BatchNorm recalibrated), mean over seeds / seed pairs:")

for comparison, g in reference.groupby("comparison"):
    per_method = g.groupby("method")["cka_bn_embedding"].mean()
    report.append(f"  {comparison:38s} " + "  ".join(f"{METHOD_NAMES[m]} {per_method[m]:.3f}" for m in SSL_METHODS))

report.append("")
write_text(OUT_DIR / "drift.txt", report)
print("\n".join(report))


# figure: embedding CKA vs label fraction, one line per encoder LR, one panel per protocol
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(4.8, 2.2), sharey=True)
x_pos = np.arange(len(FRACTIONS))
colors = {1e-4: "#56B4E9", 1e-3: "#0072B2", 1e-2: "#000000"}
random_ref = reference.loc[reference["comparison"] == "pretrained_vs_random_init", "cka_bn_embedding"].mean()

for ax, p in zip(axes, PROTOCOLS):
    for lr in ENCODER_LRS:
        g = drift[(drift["protocol"] == p) & np.isclose(drift["encoder_lr"], lr)].groupby("fraction")["cka_bn_embedding"]
        ax.errorbar(x_pos, g.mean().loc[FRACTIONS], yerr=g.std().loc[FRACTIONS], color=colors[lr], marker="o", ms=3,
                    lw=1.0, capsize=1.5, elinewidth=0.8, label=f"encoder LR {lr:.0e}")

    if np.isfinite(random_ref):
        ax.axhline(random_ref, color="0.5", lw=0.8, ls="--", label="pretrained vs random init")

    ax.set_title(f"({'ab'[PROTOCOLS.index(p)]}) {p.capitalize()} fine-tuning")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([pct(f) for f in FRACTIONS])
    ax.set_xlabel("Labeled training fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

axes[0].set_ylabel("CKA to pretrained encoder")
handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(l) for l in labels if l != "pretrained vs random init"] + [labels.index(l) for l in labels if l == "pretrained vs random init"]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.13))
fig.tight_layout(rect=(0, 0, 1, 0.92))

for ext in ("pdf", "png"):
    fig.savefig(OUT_DIR / f"fig_drift.{ext}", dpi=300, bbox_inches="tight")

print(f"wrote drift_runs.csv, drift_reference.csv, drift.txt, fig_drift.pdf/.png to {OUT_DIR}")
