# Step 6: compute cost of pretraining and downstream training

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from common import (
    FRACTIONS, METHOD_NAMES, PROTOCOLS, SSL_METHODS,
    find_sessions, load_runs, read_json, write_text
)

"""
python 06_compute.py <runs-dir> <out-dir>

all numbers are measured on the GPU each run used (see compute.json / pretrain_compute.json):
    wall-clock time, peak GPU memory allocated by PyTorch, GPU board energy from NVML
FLOPs are counted for one forward pass (torch FlopCounterMode, FLOPs = 2 x multiply-accumulates). total
pretraining compute is approximated as 3 x forward FLOPs x training samples (forward + backward); for BYOL this
slightly overestimates, because the target network is never backpropagated.

reads:
    <out-dir>/runs.csv  (from 01_collect_runs.py)
    <runs-dir>/<method>_seed<seed>/pretrain/pretrain_compute.json
    <runs-dir>/<method>_seed<seed>/<task>/downstream_model_stats.json
writes:
    <out-dir>/compute_pretrain.csv, compute_downstream.csv, table_compute.tex, compute.txt
"""


USAGE = """Usage:
    python 06_compute.py <runs-dir> <out-dir>
        runs-dir: folder containing the <method>_seed<seed>/ session folders
        out-dir:  folder with runs.csv from 01_collect_runs.py; outputs are written here
"""

if len(sys.argv) != 3:
    raise ValueError(USAGE)

RUNS_DIR, OUT_DIR = Path(sys.argv[1]), Path(sys.argv[2])
TASK = "diagnostic"

PRETRAIN_BATCH_SIZE = 128


def pct(frac):
    return f"{int(round(frac * 100))}%"


def mean_sd(s, fmt="{:.2f}"):
    s = s.dropna()
    if len(s) == 0:
        return "n/a"
    return fmt.format(s.mean()) + (f" +- {fmt.format(s.std())}" if len(s) > 1 else "")


runs = load_runs(OUT_DIR)
sessions = find_sessions(RUNS_DIR)


# pretraining, one row per objective and seed
pre_rows = []

for method, seed, path in sessions:
    if method not in SSL_METHODS:
        continue

    c = read_json(path / "pretrain" / "pretrain_compute.json")
    stats = c.get("model_stats", {})
    fwd = stats.get("forward_flops_per_training_sample")
    samples = c["n_epochs_run"] * c["steps_per_epoch"] * PRETRAIN_BATCH_SIZE

    pre_rows.append({
        "method": method,
        "seed": seed,
        "gpu_name": c.get("hardware", {}).get("gpu_name"),
        "epochs": c["n_epochs_run"],
        "wall_clock_h": c["wall_clock_s"] / 3600,
        "epoch_time_s": c.get("epoch_time_s_median_excl_first"),
        "peak_memory_gb": c.get("peak_memory_allocated_mb", np.nan) / 1024,
        "energy_kwh": c["energy_j"] / 3.6e6 if c.get("energy_j") is not None else np.nan,
        "mean_power_w": c.get("mean_power_w"),
        "params_trainable_m": stats.get("params_used_trainable", np.nan) / 1e6,
        "params_total_m": stats.get("params_used_total", np.nan) / 1e6,
        "forward_gflops_per_sample": fwd / 1e9 if fwd else np.nan,
        "train_pflops_approx": 3 * fwd * samples / 1e15 if fwd else np.nan,
    })

pre = pd.DataFrame(pre_rows)
pre.to_csv(OUT_DIR / "compute_pretrain.csv", index=False)


# downstream, one row per run (already in runs.csv)
down = runs.assign(
    minutes=runs["wall_clock_s"] / 60,
    peak_memory_gb=runs["peak_memory_allocated_mb"] / 1024,
    energy_wh=runs["energy_j"] / 3600,
    trainable_params_k=(runs["n_trainable_params_encoder"] + runs["n_trainable_params_head"]) / 1e3,
    arm=np.where(runs["method"] == "baseline", "scratch " + runs["protocol"], "SSL " + runs["protocol"]),
)
down_summary = (
    down.groupby(["arm", "fraction"])
    .agg(n_runs=("minutes", "size"), minutes=("minutes", "mean"), peak_memory_gb=("peak_memory_gb", "mean"),
         energy_wh=("energy_wh", "mean"), trainable_params_k=("trainable_params_k", "mean"))
    .reset_index()
)
down_summary.to_csv(OUT_DIR / "compute_downstream.csv", index=False)

stats_path = next((p / TASK / "downstream_model_stats.json" for _, _, p in sessions
                   if (p / TASK / "downstream_model_stats.json").exists()), None)
model_stats = read_json(stats_path) if stats_path else {}


# text report
report = ["pretraining, per objective (mean +- SD over seeds)"]
report.append(f"{'':12s}{'time (h)':>16s}{'s/epoch':>16s}{'peak mem (GB)':>16s}{'energy (kWh)':>16s}"
              f"{'power (W)':>14s}{'trainable (M)':>15s}{'fwd GFLOPs':>12s}{'train PFLOPs':>14s}")

for m in SSL_METHODS:
    g = pre[pre["method"] == m]

    if len(g) == 0:
        continue

    report.append(
        f"{METHOD_NAMES[m]:12s}{mean_sd(g['wall_clock_h']):>16s}{mean_sd(g['epoch_time_s'], '{:.1f}'):>16s}"
        f"{mean_sd(g['peak_memory_gb']):>16s}{mean_sd(g['energy_kwh'], '{:.3f}'):>16s}"
        f"{g['mean_power_w'].mean():>14.1f}{g['params_trainable_m'].mean():>15.2f}"
        f"{g['forward_gflops_per_sample'].mean():>12.3f}{g['train_pflops_approx'].mean():>14.1f}"
    )

report += ["", "downstream, mean per run (over objectives, seeds and encoder LRs)"]
report.append(f"{'':16s}" + "".join(f"{pct(f):>22s}" for f in FRACTIONS))
report.append(f"{'':16s}" + "".join(f"{'min | GB | Wh':>22s}" for _ in FRACTIONS))
arms = ["scratch full"] + [f"SSL {p}" for p in PROTOCOLS]

for arm in arms:
    d = down_summary[down_summary["arm"] == arm].set_index("fraction")

    if len(d) == 0:
        continue

    cells = [f"{d.loc[f, 'minutes']:.1f} | {d.loc[f, 'peak_memory_gb']:.2f} | {d.loc[f, 'energy_wh']:.1f}" for f in FRACTIONS]
    report.append(f"{arm:16s}" + "".join(f"{c:>22s}" for c in cells))

report.append("trainable parameters per downstream run: "
              + ", ".join(f"{arm} {down_summary.loc[down_summary['arm'] == arm, 'trainable_params_k'].mean():.0f}k"
                          for arm in arms if arm in set(down_summary["arm"])))

if model_stats:
    report.append(
        f"downstream network: encoder {model_stats['params_encoder'] / 1e6:.2f}M parameters, "
        f"last stage {model_stats['params_last_stage'] / 1e3:.0f}k, head {model_stats['params_head'] / 1e3:.1f}k, "
        f"forward {model_stats['forward_flops_per_crop'] / 1e6:.1f} MFLOPs per 2.5 s crop"
    )

pre_h, pre_kwh = pre["wall_clock_h"].sum(), pre["energy_kwh"].sum()
down_h, down_kwh = down["wall_clock_s"].sum() / 3600, down["energy_j"].sum() / 3.6e6
per_session = down[down["method"] != "baseline"].groupby(["method", "seed"])["wall_clock_s"].sum() / 3600

report += [
    "",
    "totals (GPU time inside the training loops; excludes data loading and notebook overhead):",
    f"  pretraining: {len(pre)} runs, {pre_h:.1f} GPU-h, {pre_kwh:.2f} kWh",
    f"  downstream:  {len(down)} runs, {down_h:.1f} GPU-h, {down_kwh:.2f} kWh",
    f"  whole study: {pre_h + down_h:.1f} GPU-h, {pre_kwh + down_kwh:.2f} kWh",
    f"  one SSL objective and seed: pretraining {pre['wall_clock_h'].mean():.2f} h + downstream grid "
    f"{per_session.mean():.2f} h = {pre['wall_clock_h'].mean() + per_session.mean():.2f} h",
    f"  GPUs: {', '.join(sorted(set(map(str, pd.concat([pre['gpu_name'], down['gpu_name']]).dropna()))))}"
    f" | energy measured by: {', '.join(sorted(set(map(str, down['energy_method'].dropna()))))}",
]

write_text(OUT_DIR / "compute.txt", report)
print("\n".join(report))


# LaTeX table: pretraining cost per objective
def latex_table():
    lines = [
        "% requires \\usepackage{booktabs}",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pretraining cost per objective (100 epochs, mean over "
        f"{pre['seed'].nunique()} seeds, one {pre['gpu_name'].dropna().iloc[0] if pre['gpu_name'].notna().any() else 'GPU'}). "
        "Parameters used during pretraining, forward FLOPs per training sample, wall-clock time, peak GPU memory, "
        "and GPU energy measured with NVML.}",
        "\\label{tab:compute}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Objective & Params (M) & GFLOPs & Time (h) & Memory (GB) & Energy (kWh) \\\\",
        "\\midrule",
    ]

    for m in SSL_METHODS:
        g = pre[pre["method"] == m]

        if len(g) == 0:
            continue

        lines.append(
            f"{METHOD_NAMES[m]} & {g['params_trainable_m'].mean():.2f} & {g['forward_gflops_per_sample'].mean():.2f} & "
            f"{g['wall_clock_h'].mean():.2f} & {g['peak_memory_gb'].mean():.2f} & {g['energy_kwh'].mean():.2f} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    return lines


write_text(OUT_DIR / "table_compute.tex", latex_table())
print(f"wrote compute_pretrain.csv, compute_downstream.csv, table_compute.tex, compute.txt to {OUT_DIR}")
