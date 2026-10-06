# Step 4: Secondary metrics on the selected runs

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from common import (
    FRACTIONS, METHOD_NAMES, METHODS, PROTOCOL_NAMES, PROTOCOLS, frac_tag, write_text
)

"""
python 04_secondary_metrics.py <runs-dir> <out-dir>

    1. pooling sensitivity: record-level scores from max-probability (used for training and model selection),
       mean-probability and mean-logit pooling over the 7 chunks, on the same checkpoints
    2. calibration: classwise expected calibration error (15 equal-width bins per label, averaged over labels)
    3. per-label test AUROC and AUPRC (max-probability pooling)
    4. label-support sensitivity: macro AUROC / AUPRC over labels with >= MIN_TEST_POS test positives, and over
       labels with >= MIN_TRAIN_POS positives in that seed's labeled subset

reads:
    <out-dir>/selected_runs.csv                  (from 02_select_and_tabulate.py)
    <runs-dir>/.../test_logits.npy, labels.npz, label_counts.csv
writes:
    <out-dir>/secondary_runs.csv   one row per selected run
    <out-dir>/per_label.csv        per-label results, mean and SD over seeds
    <out-dir>/secondary.txt        summary tables (also printed)
"""


USAGE = """Usage:
    python 04_secondary_metrics.py <runs-dir> <out-dir>
        runs-dir: folder containing the <method>_seed<seed>/ session folders
        out-dir:  folder with selected_runs.csv from 02_select_and_tabulate.py; outputs are written here
"""

if len(sys.argv) != 3:
    raise ValueError(USAGE)

RUNS_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
TASK = "diagnostic"

MIN_TEST_POS = 20
MIN_TRAIN_POS = 5
N_BINS = 15
POOLINGS = ["max_prob", "mean_prob", "mean_logit"]


def sigmoid(x):
    with np.errstate(over="ignore"):
        return (1.0 / (1.0 + np.exp(-x.astype(np.float32)))).astype(np.float32)


def pool(logits, rule):
    # (N, K, C) chunk logits -> (N, C) record-level probabilities

    if rule == "max_prob":
        return sigmoid(logits).max(axis=1)

    if rule == "mean_prob":
        return sigmoid(logits).mean(axis=1)

    return sigmoid(logits.mean(axis=1))


def per_class(metric_fn, y, p):
    # nan for classes without both positives and negatives (same rule as safe_macro_auc)
    return np.array([metric_fn(y[:, c], p[:, c]) if 0 < y[:, c].sum() < len(y) else np.nan for c in range(y.shape[1])])


def classwise_ece(y, p, n_bins=N_BINS):
    # per-label ECE with equal-width bins; returns (C,) array

    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = np.zeros(y.shape[1])

    for b in range(n_bins):
        in_bin = bins == b
        n = in_bin.sum(axis=0)
        gap = np.abs((y * in_bin).sum(axis=0) - (p * in_bin).sum(axis=0))
        ece += np.where(n > 0, gap, 0.0) / len(y)

    return ece


def pct(frac):
    return f"{int(round(frac * 100))}%"


def row_order():
    rows = [("baseline", "full")]

    for m in METHODS[1:]:
        rows += [(m, p) for p in PROTOCOLS]

    return rows


def row_label(method, protocol):
    return METHOD_NAMES[method] if method == "baseline" else f"{METHOD_NAMES[method]} {PROTOCOL_NAMES[protocol]}"


def text_table(df, col, title, fmt="{:.3f}"):
    # mean over seeds of `col`, rows = method/protocol, columns = label fraction

    means = df.groupby(["method", "protocol", "fraction"])[col].mean()
    lines = [title, f"{'':20s}" + "".join(f"{pct(f):>9s}" for f in FRACTIONS)]

    for m, p in row_order():
        lines.append(f"{row_label(m, p):20s}" + "".join(f"{fmt.format(means.loc[(m, p, f)]):>9s}" for f in FRACTIONS))

    return lines + [""]


selected = pd.read_csv(OUT_DIR / "selected_runs.csv")
seeds = sorted(int(s) for s in selected["seed"].unique())

with np.load(RUNS_DIR / f"baseline_seed{seeds[0]}" / TASK / "labels.npz", allow_pickle=False) as z:
    y_test = z["y_test"].astype(np.float32)
    label_names = [str(n) for n in z["label_names"]]

test_pos = y_test.sum(axis=0)
supported = test_pos >= MIN_TEST_POS


# training positives per label in each seed's labeled subsets (identical across methods, see audit)
train_pos = {s: pd.read_csv(RUNS_DIR / f"baseline_seed{s}" / TASK / "label_counts.csv") for s in seeds}

rows, label_rows = [], []

print(f"evaluating {len(selected)} selected runs ...")

for r in selected.itertuples():
    run_dir = RUNS_DIR / f"{r.method}_seed{r.seed}" / TASK / r.config / frac_tag(r.fraction)
    logits = np.load(run_dir / "test_logits.npy")
    out = {"method": r.method, "protocol": r.protocol, "fraction": r.fraction, "seed": r.seed}

    for rule in POOLINGS:
        p = pool(logits, rule)
        auc, ap = per_class(roc_auc_score, y_test, p), per_class(average_precision_score, y_test, p)
        out[f"auroc_{rule}"], out[f"auprc_{rule}"] = np.nanmean(auc), np.nanmean(ap)
        out[f"ece_{rule}"] = classwise_ece(y_test, p)[test_pos > 0].mean()

        if rule == "max_prob":
            enough_train = train_pos[r.seed][f"train_pos_{frac_tag(r.fraction)}"].values >= MIN_TRAIN_POS
            out["auroc_test_supported"] = np.nanmean(auc[supported])
            out["auprc_test_supported"] = np.nanmean(ap[supported])
            out["auroc_train_supported"] = np.nanmean(auc[enough_train]) if enough_train.any() else np.nan
            out["auprc_train_supported"] = np.nanmean(ap[enough_train]) if enough_train.any() else np.nan
            out["n_labels_train_supported"] = int(enough_train.sum())

            label_rows += [
                {"method": r.method, "protocol": r.protocol, "fraction": r.fraction, "seed": r.seed,
                 "label": name, "auroc": a, "auprc": b}
                for name, a, b in zip(label_names, auc, ap)
            ]

    rows.append(out)

runs = pd.DataFrame(rows)
runs.to_csv(OUT_DIR / "secondary_runs.csv", index=False)

per_label = (
    pd.DataFrame(label_rows)
    .groupby(["method", "protocol", "fraction", "label"])
    .agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"))
    .reset_index()
)
per_label["test_pos"] = per_label["label"].map(dict(zip(label_names, test_pos.astype(int))))
per_label.to_csv(OUT_DIR / "per_label.csv", index=False)


# pooling: do rankings and conclusions change?
means = runs.groupby(["method", "protocol", "fraction"])[[f"auroc_{r}" for r in POOLINGS]].mean()
scratch = runs[runs["method"] == "baseline"].set_index(["seed", "fraction"])

report = [f"seeds {seeds}, {len(y_test)} test records, {len(label_names)} labels", ""]
report += ["pooling sensitivity (same checkpoints; selection used max_prob)"]

for rule in POOLINGS[1:]:
    rho = [means.xs(f, level="fraction")[["auroc_max_prob", f"auroc_{rule}"]].corr(method="spearman").iloc[0, 1]
           for f in FRACTIONS]

    flips = 0

    for (m, p, f), g in runs[runs["method"] != "baseline"].groupby(["method", "protocol", "fraction"]):
        d_max = np.mean([x.auroc_max_prob - scratch.loc[(x.seed, f), "auroc_max_prob"] for x in g.itertuples()])
        d_rule = np.mean([getattr(x, f"auroc_{rule}") - scratch.loc[(x.seed, f), f"auroc_{rule}"] for x in g.itertuples()])
        flips += int(np.sign(d_max) != np.sign(d_rule))

    shift = (means[f"auroc_{rule}"] - means["auroc_max_prob"])
    report.append(
        f"  {rule:10s} vs max_prob: Spearman rho of the 13 rows per fraction "
        + ", ".join(f"{pct(f)} {r_:.2f}" for f, r_ in zip(FRACTIONS, rho))
        + f" | macro AUROC shift mean {shift.mean():+.4f} (range {shift.min():+.4f} to {shift.max():+.4f})"
        + f" | sign flips of SSL-minus-scratch: {flips}/60"
    )

report.append("")
report += text_table(runs, "auroc_mean_prob", "test macro AUROC, mean-probability pooling")
report += text_table(runs, "ece_max_prob", "classwise ECE, max-probability pooling (lower is better)")
report += text_table(runs, "ece_mean_prob", "classwise ECE, mean-probability pooling (lower is better)")

report.append(f"{int(supported.sum())}/{len(label_names)} labels have >= {MIN_TEST_POS} test positives "
              f"(test positives per label: min {int(test_pos.min())}, median {int(np.median(test_pos))})")
report.append(f"chance-level macro AUPRC (mean test prevalence over labels with positives): "
              f"{(test_pos[test_pos > 0] / len(y_test)).mean():.3f}")
report.append("")
report += text_table(runs, "auroc_test_supported", f"test macro AUROC over labels with >= {MIN_TEST_POS} test positives")
report += text_table(runs, "auprc_test_supported", f"test macro AUPRC over labels with >= {MIN_TEST_POS} test positives")

n_train = runs[runs["method"] == "baseline"].groupby("fraction")["n_labels_train_supported"].agg(["min", "max"])
report.append(f"labels with >= {MIN_TRAIN_POS} training positives in the subset (min-max over seeds): "
              + ", ".join(f"{pct(f)} {n_train.loc[f, 'min']}-{n_train.loc[f, 'max']}" for f in FRACTIONS))
report.append("")
report += text_table(runs, "auroc_train_supported", f"test macro AUROC over labels with >= {MIN_TRAIN_POS} training positives")

write_text(OUT_DIR / "secondary.txt", report)
print("\n".join(report))
print(f"wrote secondary_runs.csv, per_label.csv, secondary.txt to {OUT_DIR}")
