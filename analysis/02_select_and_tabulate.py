# Step 2: Select the encoder learning rate on validation, and build the results tables

import sys
from pathlib import Path
import numpy as np
from common import (
    FRACTIONS, METHOD_NAMES, METHODS, PROTOCOL_NAMES, PROTOCOLS, load_runs, write_text
)

"""
python 02_select_and_tabulate.py <out-dir>

selection rule (fixed before looking at test):  for every method, protocol and label fraction, 
                                                the encoder LR with the highest validation macro AUROC averaged over seeds.
                                                one LR is chosen per cell and shared by all seeds.

reads:
    <out-dir>/runs.csv            (from 01_collect_runs.py)
writes:
    <out-dir>/selection.csv       mean validation AUROC of every encoder LR, and the selected one
    <out-dir>/selected_runs.csv   the selected run of every method / protocol / fraction / seed (input to steps 3-4)
    <out-dir>/grid.csv            mean and SD over seeds for every configuration, selected or not
    <out-dir>/table_main.tex      test macro AUROC and AUPRC, mean +- SD, selected configurations
    <out-dir>/results.txt         the same tables as text (also printed)
"""


USAGE = """Usage:
    python 02_select_and_tabulate.py <out-dir>
        out-dir: folder with runs.csv from 01_collect_runs.py; outputs are written here
"""

if len(sys.argv) != 2:
    raise ValueError(USAGE)

OUT_DIR = Path(sys.argv[1])

runs = load_runs(OUT_DIR)


def row_order():
    # (method, protocol) rows in table order: scratch first, then each objective's protocols

    rows = [("baseline", "full")]

    for m in METHODS[1:]:
        rows += [(m, p) for p in PROTOCOLS]

    return rows


def row_label(method, protocol):
    return METHOD_NAMES[method] if method == "baseline" else f"{METHOD_NAMES[method]} {PROTOCOL_NAMES[protocol]}"


def pct(frac):
    return f"{int(round(frac * 100))}%"


# selection: highest validation AUROC averaged over seeds
mean_val = (
    runs.groupby(["method", "protocol", "fraction", "encoder_lr"])["best_val_macro_auroc"]
    .agg(["mean", "count"])
    .reset_index()
)

best = mean_val.loc[mean_val.groupby(["method", "protocol", "fraction"])["mean"].idxmax()]
best = best.rename(columns={"encoder_lr": "selected_encoder_lr", "mean": "selected_mean_val_auroc"})

selection = mean_val.pivot_table(index=["method", "protocol", "fraction"], columns="encoder_lr", values="mean")
selection.columns = [f"mean_val_auroc_enc{lr:.0e}" for lr in selection.columns]
selection = selection.reset_index().merge(
    best[["method", "protocol", "fraction", "selected_encoder_lr", "selected_mean_val_auroc"]],
    on=["method", "protocol", "fraction"],
)
selection.to_csv(OUT_DIR / "selection.csv", index=False)

selected = runs.merge(
    best[["method", "protocol", "fraction", "selected_encoder_lr"]], on=["method", "protocol", "fraction"]
)
selected = selected[np.isclose(selected["encoder_lr"], selected["selected_encoder_lr"])].drop(columns="selected_encoder_lr")
selected.to_csv(OUT_DIR / "selected_runs.csv", index=False)


# every configuration, mean and SD over seeds
grid = (
    runs.groupby(["method", "protocol", "encoder_lr", "fraction"])
    .agg(
        n_seeds=("seed", "nunique"),
        val_auroc_mean=("best_val_macro_auroc", "mean"),
        test_auroc_mean=("test_macro_auroc", "mean"),
        test_auroc_sd=("test_macro_auroc", "std"),
        test_auprc_mean=("test_macro_auprc", "mean"),
        test_auprc_sd=("test_macro_auprc", "std"),
        final_train_loss_mean=("final_train_loss", "mean"),
    )
    .reset_index()
)
grid.to_csv(OUT_DIR / "grid.csv", index=False)


# summary of the selected configurations
summary = (
    selected.groupby(["method", "protocol", "fraction"])
    .agg(
        auroc_mean=("test_macro_auroc", "mean"),
        auroc_sd=("test_macro_auroc", "std"),
        auprc_mean=("test_macro_auprc", "mean"),
        auprc_sd=("test_macro_auprc", "std"),
        loss_mean=("final_train_loss", "mean"),
        n_seeds=("seed", "nunique"),
    )
    .reset_index()
    .set_index(["method", "protocol", "fraction"])
)


# paired difference to scratch: same seed means same labeled subset, head initialization and batch order
scratch = selected[selected["method"] == "baseline"].set_index(["seed", "fraction"])

paired = selected[selected["method"] != "baseline"].copy()
paired["delta_auroc"] = [
    r.test_macro_auroc - scratch.loc[(r.seed, r.fraction), "test_macro_auroc"] for r in paired.itertuples()
]
paired_summary = (
    paired.groupby(["method", "protocol", "fraction"])["delta_auroc"]
    .agg(mean="mean", n_positive=lambda d: int((d > 0).sum()), n="count")
)


# text tables
def text_table(metric, title, fmt="{:.3f} +- {:.3f}"):
    lines = [title, f"{'':20s}" + "".join(f"{pct(f):>17s}" for f in FRACTIONS)]

    for m, p in row_order():
        cells = []

        for f in FRACTIONS:
            s = summary.loc[(m, p, f)]
            cells.append(fmt.format(s[f"{metric}_mean"], s[f"{metric}_sd"]) if metric != "loss" else f"{s['loss_mean']:.3f}")

        lines.append(f"{row_label(m, p):20s}" + "".join(f"{c:>17s}" for c in cells))

    return lines


def lr_table():
    lines = ["selected encoder LR (frozen has none)", f"{'':20s}" + "".join(f"{pct(f):>9s}" for f in FRACTIONS)]
    sel = best.set_index(["method", "protocol", "fraction"])["selected_encoder_lr"]

    for m, p in row_order():
        if p == "frozen":
            continue

        lines.append(f"{row_label(m, p):20s}" + "".join(f"{sel.loc[(m, p, f)]:>9.0e}" for f in FRACTIONS))

    return lines


def delta_table():
    lines = [
        "test AUROC minus scratch, mean of per-seed paired differences [seeds with a positive difference]",
        f"{'':20s}" + "".join(f"{pct(f):>15s}" for f in FRACTIONS),
    ]

    for m, p in row_order()[1:]:
        cells = [paired_summary.loc[(m, p, f)] for f in FRACTIONS]
        lines.append(
            f"{row_label(m, p):20s}" + "".join(f"{c['mean']:>+9.3f} [{int(c['n_positive'])}/{int(c['n'])}]" for c in cells)
        )

    return lines


seeds = sorted(int(s) for s in runs["seed"].unique())
report = [f"seeds: {seeds}", ""]
report += text_table("auroc", "test macro AUROC, mean +- SD over seeds (selected encoder LR)") + [""]
report += text_table("auprc", "test macro AUPRC, mean +- SD over seeds (selected encoder LR)") + [""]
report += delta_table() + [""]
report += lr_table() + [""]
report += text_table("loss", "final training loss, mean over seeds (selected encoder LR)") + [""]


# LaTeX table: AUROC and AUPRC, best mean per column in bold
def latex_table():
    def cell(metric, m, p, f, best_value):
        s = summary.loc[(m, p, f)]
        text = f"{s[metric + '_mean']:.3f} $\\pm$ {s[metric + '_sd']:.3f}"
        return f"\\textbf{{{text}}}" if round(s[metric + "_mean"], 3) == round(best_value, 3) else text

    lines = [
        "% requires \\usepackage{booktabs} and \\usepackage{multirow}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Test macro AUROC and macro AUPRC on PTB-XL diagnostic classification (44 labels), mean $\\pm$ "
        f"standard deviation over {len(seeds)} seeds. For partial and full fine-tuning, the encoder learning rate "
        "is selected per label fraction by mean validation macro AUROC. The best mean in each column (to three decimals) is in bold.}",
        "\\label{tab:results}",
        "\\begin{tabular}{ll" + "c" * len(FRACTIONS) + "}",
        "\\toprule",
    ]

    for metric, name in (("auroc", "Macro AUROC"), ("auprc", "Macro AUPRC")):
        col_best = {f: max(summary.loc[(m, p, f)][metric + "_mean"] for m, p in row_order()) for f in FRACTIONS}

        lines.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{name}}}}} & " + " & ".join(pct(f).replace("%", "\\%") for f in FRACTIONS) + " \\\\")
        lines.append("\\midrule")
        lines.append("Scratch & Full & " + " & ".join(cell(metric, "baseline", "full", f, col_best[f]) for f in FRACTIONS) + " \\\\")

        for m in METHODS[1:]:
            lines.append("\\midrule")

            for i, p in enumerate(PROTOCOLS):
                name_cell = f"\\multirow{{3}}{{*}}{{{METHOD_NAMES[m]}}}" if i == 0 else ""
                lines.append(
                    f"{name_cell} & {PROTOCOL_NAMES[p]} & "
                    + " & ".join(cell(metric, m, p, f, col_best[f]) for f in FRACTIONS)
                    + " \\\\"
                )

        lines.append("\\midrule" if metric == "auroc" else "\\bottomrule")

    lines += ["\\end{tabular}", "\\end{table*}"]

    return lines


write_text(OUT_DIR / "table_main.tex", latex_table())
write_text(OUT_DIR / "results.txt", report)
print("\n".join(report))
print(f"wrote selection.csv, selected_runs.csv, grid.csv, table_main.tex, results.txt to {OUT_DIR}")
