# Step 3: Paired bootstrap confidence intervals for the pre-specified comparisons, and the delta-vs-scratch figure

import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from common import (
    FRACTIONS, METHOD_NAMES, PROTOCOL_NAMES, PROTOCOLS, SSL_METHODS,
    find_sessions, frac_tag, write_text
)

"""
python 03_bootstrap_stats.py <runs-dir> <out-dir>

comparisons (selected encoder LR, every label fraction, macro AUROC and macro AUPRC):
    A. every SSL objective and protocol minus the scratch baseline
    B. full fine-tuning minus frozen, per SSL objective

the difference is averaged over seeds; seeds are paired (same labeled subset, head initialization, batch order).
every model is evaluated on the same bootstrap resamples of the test records. two intervals are reported:
    test: resample test records only (uncertainty from the finite test set, given the trained models)
    hier: resample seeds and test records (also includes training variability); used for p-values
p-values are two-sided, from a normal approximation with the hierarchical bootstrap standard error (a resample
count cannot resolve p-values below 1/N_BOOT, which Holm correction over 80 comparisons would need), and are
Holm-corrected within each metric over all comparisons.

reads:
    <out-dir>/selected_runs.csv    (from 02_select_and_tabulate.py)
    <runs-dir>/.../test_logits.npy, labels.npz
writes:
    <out-dir>/stats.csv, stats.txt
    <out-dir>/fig_delta_auroc.pdf/.png, fig_delta_auprc.pdf/.png
"""


USAGE = """Usage:
    python 03_bootstrap_stats.py <runs-dir> <out-dir>
        runs-dir: folder containing the <method>_seed<seed>/ session folders
        out-dir:  folder with selected_runs.csv from 02_select_and_tabulate.py; outputs are written here
"""

if len(sys.argv) != 3:
    raise ValueError(USAGE)

RUNS_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
TASK = "diagnostic"

N_BOOT = 1000
CHUNK = 50  # bootstrap resamples processed at once (memory ~ CHUNK x labels x records x 4 bytes x 6)
BOOT_SEED = 0
ALPHA = 0.05


def tie_groups(sorted_scores):
    # for every position of an ascending-sorted (C, N) array: first and last index of its group of equal scores

    n = sorted_scores.shape[1]
    idx = np.arange(n)

    first = np.ones_like(sorted_scores, dtype=bool)
    first[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    last = np.ones_like(first)
    last[:, :-1] = first[:, 1:]

    start = np.maximum.accumulate(np.where(first, idx, 0), axis=1)
    end = np.minimum.accumulate(np.where(last, idx, n - 1)[:, ::-1], axis=1)[:, ::-1]

    return start, end


def bootstrap_macro(scores, y, weights):
    """
    macro AUROC and macro AUPRC on weighted (resampled) test sets, with ties handled as in scikit-learn
    (a tied positive/negative pair counts 1/2 for AUROC; tied scores form one threshold for average precision).

    scores:  (N, C) record-level probabilities
    y:       (N, C) binary labels
    weights: (B, N) number of times each record is drawn in each resample
    returns: auroc (B,), auprc (B,); a class enters a resample's macro average only if that resample
             contains both positives and negatives for it (same rule as safe_macro_auc)
    """

    order = np.argsort(scores, axis=0, kind="mergesort").T                     # (C, N) ascending score
    y_sorted = np.take_along_axis(y, order.T, axis=0).T.astype(np.float32)     # (C, N)
    start, end = tie_groups(np.take_along_axis(scores, order.T, axis=0).T)
    start, end1 = start[None], end[None] + 1                                   # indices into zero-padded cumsums
    auroc, auprc = [], []

    for b in range(0, len(weights), CHUNK):
        w = weights[b:b + CHUNK].astype(np.float32)[:, order]                  # (b, C, N)
        pos = w * y_sorted
        neg = w - pos
        n_pos, n_neg = pos.sum(-1), neg.sum(-1)
        valid = (n_pos > 0) & (n_neg > 0)

        zero = np.zeros(pos.shape[:-1] + (1,), dtype=np.float32)
        cum_pos = np.concatenate([zero, np.cumsum(pos, axis=-1)], axis=-1)    # cum_pos[..., i] = sum of pos[..., :i]
        cum_neg = np.concatenate([zero, np.cumsum(neg, axis=-1)], axis=-1)

        neg_below = np.take_along_axis(cum_neg, start, axis=-1)
        neg_tied = np.take_along_axis(cum_neg, end1, axis=-1) - neg_below
        pos_below = np.take_along_axis(cum_pos, start, axis=-1)

        with np.errstate(divide="ignore", invalid="ignore"):
            auc = (pos * (neg_below + 0.5 * neg_tied)).sum(-1) / (n_pos * n_neg)

            # precision at the threshold of each positive's tie group: everything scored at or above it
            tp = n_pos[..., None] - pos_below
            fp = n_neg[..., None] - neg_below
            ap = (pos * tp / np.maximum(tp + fp, 1e-12)).sum(-1) / n_pos

        auroc.append(np.nanmean(np.where(valid, auc, np.nan), axis=1))
        auprc.append(np.nanmean(np.where(valid, ap, np.nan), axis=1))

    return np.concatenate(auroc), np.concatenate(auprc)


def record_scores(logits):
    # (N, K, C) chunk logits -> (N, C) record probabilities: float32 sigmoid, then max over chunks (as in training)
    with np.errstate(over="ignore"):
        probs = (1.0 / (1.0 + np.exp(-logits.astype(np.float32)))).astype(np.float32)
    return probs.max(axis=1)


def holm(p):
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0

    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)

    return adjusted


def pct(frac):
    return f"{int(round(frac * 100))}%"


selected = pd.read_csv(OUT_DIR / "selected_runs.csv")
seeds = sorted(int(s) for s in selected["seed"].unique())

method, seed, path = find_sessions(RUNS_DIR)[0]

with np.load(path / TASK / "labels.npz", allow_pickle=False) as z:
    y_test = z["y_test"].astype(np.float32)

n_test = len(y_test)
rng = np.random.default_rng(BOOT_SEED)
weights = rng.multinomial(n_test, np.full(n_test, 1.0 / n_test), size=N_BOOT)      # (B, N), shared by all runs
point_weights = np.ones((1, n_test))
seed_draws = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))                # hierarchical: seeds per resample


# bootstrap distribution of every selected run
boot, point = {}, {}
max_check_diff = 0.0

print(f"bootstrapping {len(selected)} selected runs, {N_BOOT} resamples of {n_test} test records ...")

for i, r in enumerate(selected.itertuples()):
    run_dir = RUNS_DIR / f"{r.method}_seed{r.seed}" / TASK / r.config / frac_tag(r.fraction)
    scores = record_scores(np.load(run_dir / "test_logits.npy"))

    key = (r.method, r.protocol, r.fraction, r.seed)
    boot[key] = bootstrap_macro(scores, y_test, weights)
    point[key] = tuple(v[0] for v in bootstrap_macro(scores, y_test, point_weights))
    max_check_diff = max(max_check_diff, abs(point[key][0] - r.test_macro_auroc))

    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(selected)}")

print(f"check: max |recomputed - stored| test macro AUROC = {max_check_diff:.1e}")

if max_check_diff > 1e-3:
    raise RuntimeError("recomputed AUROC does not match metrics.json; check the run folders")


# comparisons
comparisons = [("vs_scratch", (m, p), ("baseline", "full")) for m in SSL_METHODS for p in PROTOCOLS]
comparisons += [("full_vs_frozen", (m, "full"), (m, "frozen")) for m in SSL_METHODS]

rows = []

for family, (ma, pa), (mb, pb) in comparisons:
    for frac in FRACTIONS:
        for k, metric in enumerate(("auroc", "auprc")):
            per_seed = np.array([point[(ma, pa, frac, s)][k] - point[(mb, pb, frac, s)][k] for s in seeds])
            d_boot = np.stack([boot[(ma, pa, frac, s)][k] - boot[(mb, pb, frac, s)][k] for s in seeds], axis=1)

            d_test = d_boot.mean(axis=1)
            d_hier = np.take_along_axis(d_boot, seed_draws, axis=1).mean(axis=1)
            se = d_hier.std(ddof=1)
            p = math.erfc(abs(per_seed.mean()) / (se * math.sqrt(2))) if se > 0 else 0.0

            rows.append({
                "family": family,
                "metric": metric,
                "method": ma,
                "protocol": pa,
                "reference": f"{mb}_{pb}",
                "fraction": frac,
                "delta": per_seed.mean(),
                "ci_test_lo": np.quantile(d_test, ALPHA / 2),
                "ci_test_hi": np.quantile(d_test, 1 - ALPHA / 2),
                "ci_hier_lo": np.quantile(d_hier, ALPHA / 2),
                "ci_hier_hi": np.quantile(d_hier, 1 - ALPHA / 2),
                "se_hier": se,
                "p_hier": p,
                "n_seeds_positive": int((per_seed > 0).sum()),
                **{f"delta_seed{s}": d for s, d in zip(seeds, per_seed)},
            })

stats = pd.DataFrame(rows)
stats["p_holm"] = np.nan

for metric in ("auroc", "auprc"):
    idx = stats["metric"] == metric
    stats.loc[idx, "p_holm"] = holm(stats.loc[idx, "p_hier"])

stats["significant"] = stats["p_holm"] < ALPHA
stats.to_csv(OUT_DIR / "stats.csv", index=False)


# text report
def table(family, metric, title):
    s = stats[(stats["family"] == family) & (stats["metric"] == metric)].set_index(["method", "protocol", "fraction"])
    lines = [title, f"{'':18s}" + "".join(f"{pct(f):>26s}" for f in FRACTIONS)]
    protocols = PROTOCOLS if family == "vs_scratch" else ["full"]

    for m in SSL_METHODS:
        for p in protocols:
            cells = []

            for f in FRACTIONS:
                r = s.loc[(m, p, f)]
                mark = "*" if r["significant"] else " "
                cells.append(f"{r['delta']:+.3f} [{r['ci_hier_lo']:+.3f},{r['ci_hier_hi']:+.3f}]{mark}")

            label = f"{METHOD_NAMES[m]} {PROTOCOL_NAMES[p]}" if family == "vs_scratch" else METHOD_NAMES[m]
            lines.append(f"{label:18s}" + "".join(f"{c:>26s}" for c in cells))

    return lines + [""]


report = [
    f"seeds {seeds}, {N_BOOT} bootstrap resamples, {n_test} test records",
    "cells: mean paired difference [95% hierarchical CI]; * = Holm-corrected p < 0.05 within the metric",
    "",
]
report += table("vs_scratch", "auroc", "macro AUROC, SSL minus scratch")
report += table("full_vs_frozen", "auroc", "macro AUROC, full minus frozen")
report += table("vs_scratch", "auprc", "macro AUPRC, SSL minus scratch")
report += table("full_vs_frozen", "auprc", "macro AUPRC, full minus frozen")

for metric in ("auroc", "auprc"):
    s = stats[stats["metric"] == metric]
    report.append(f"{metric}: {int(s['significant'].sum())}/{len(s)} comparisons significant after Holm correction")

write_text(OUT_DIR / "stats.txt", report)
print("\n".join(report))


# figure: delta vs scratch, one panel per protocol, one line per objective
COLORS = {"mask": "#0072B2", "denoise": "#009E73", "simclr": "#D55E00", "byol": "#CC79A7"}
MARKERS = {"mask": "o", "denoise": "s", "simclr": "^", "byol": "D"}
PANEL_TITLES = {"frozen": "(a) Frozen encoder", "partial": "(b) Partial (last stage)", "full": "(c) Full fine-tuning"}


def plot_delta(metric, ylabel, filename):
    # filled markers: hierarchical 95% CI excludes zero; hollow markers: it does not

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
    s = stats[(stats["family"] == "vs_scratch") & (stats["metric"] == metric)]
    x = np.arange(len(FRACTIONS))
    offsets = np.linspace(-0.21, 0.21, len(SSL_METHODS))

    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(7.0, 2.3), sharey=True)

    for ax, protocol in zip(axes, PROTOCOLS):
        ax.axhline(0.0, color="0.5", lw=0.8, zorder=0)

        for off, m in zip(offsets, SSL_METHODS):
            d = s[(s["method"] == m) & (s["protocol"] == protocol)].set_index("fraction").loc[FRACTIONS]
            yerr = [d["delta"] - d["ci_hier_lo"], d["ci_hier_hi"] - d["delta"]]
            excludes_zero = ((d["ci_hier_lo"] > 0) | (d["ci_hier_hi"] < 0)).values

            ax.errorbar(x + off, d["delta"], yerr=yerr, color=COLORS[m], lw=1.0, capsize=1.5, elinewidth=0.8,
                        marker="none")
            ax.scatter(x + off, d["delta"], s=14, marker=MARKERS[m], edgecolors=COLORS[m], linewidths=0.9, zorder=3,
                       facecolors=[COLORS[m] if e else "white" for e in excludes_zero])

        ax.set_title(PANEL_TITLES[protocol])
        ax.set_xticks(x)
        ax.set_xticklabels([pct(f) for f in FRACTIONS])
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.grid(axis="y", lw=0.4, alpha=0.5)

    axes[0].set_ylabel(ylabel)
    axes[len(axes) // 2].set_xlabel("Labeled training fraction")
    handles = [Line2D([0], [0], color=COLORS[m], marker=MARKERS[m], ms=3.5, lw=1.0) for m in SSL_METHODS]
    fig.legend(handles, [METHOD_NAMES[m] for m in SSL_METHODS], loc="upper center", ncol=len(SSL_METHODS),
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{filename}.{ext}", dpi=300, bbox_inches="tight")

    plt.close(fig)


plot_delta("auroc", "$\\Delta$ macro AUROC vs scratch", "fig_delta_auroc")
plot_delta("auprc", "$\\Delta$ macro AUPRC vs scratch", "fig_delta_auprc")

print(f"wrote stats.csv, stats.txt, fig_delta_auroc.pdf/.png, fig_delta_auprc.pdf/.png to {OUT_DIR}")
