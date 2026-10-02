import json, os
from pathlib import Path
import numpy as np
import pandas as pd


"""
Patient-grouped, multilabel-stratified, nested label-fraction subsets.

The training folds are split into N_FOLDS (default 100) small folds with iterative stratification (Sechidis et al., 2011),
treating each patient as one indivisible item so that all records of a patient land in the same small fold.

A label fraction f is the union of the first round(f * N_FOLDS) small folds, so:
    - Every subset is stratified by the task labels
    - Subsets are nested (1% is inside 5% is inside 10% is inside 25%)
    - Subsets are patient-level (no patient is split across folds)

PTB-XL's own 10 strat_folds were built with a comparable patient-respecting stratified procedure (Wagner et al., 2020).
"""


# subsets for these fractions are always built and saved, whatever a session runs
STANDARD_FRACTIONS = (0.01, 0.05, 0.10, 0.25, 1.00)


def _choose_fold(primary, secondary, rng, tol=1e-9):
    """
    Pick the fold with the largest remaining demand for the current label (primary),
    break ties by the largest remaining demand for records overall (secondary), break any remaining ties uniformly at random.
    """
    cand = np.flatnonzero(primary >= primary.max() - tol)

    if len(cand) > 1:
        sec = secondary[cand]
        cand = cand[sec >= sec.max() - tol]

    if len(cand) > 1:
        return int(rng.choice(cand))

    return int(cand[0])


def iterative_stratified_folds(y, groups, n_folds, seed):
    """
    y:       (n_records, n_labels) binary label matrix
    groups:  (n_records,) group id per record (patient_id)
    n_folds: number of folds
    seed:    controls tie-breaking and the final fold permutation

    returns: (n_records,) fold id in [0, n_folds) for every record
    """
    
    y = np.asarray(y, dtype=np.float64)
    rng = np.random.default_rng(seed)

    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = int(inv.max()) + 1

    # label counts and record counts per group (patient)
    group_labels = np.zeros((n_groups, y.shape[1]), dtype=np.float64)
    np.add.at(group_labels, inv, y)
    group_sizes = np.bincount(inv, minlength=n_groups).astype(np.float64)

    # remaining demand per fold, overall and per label
    demand_total = np.full(n_folds, group_sizes.sum() / n_folds)
    demand_label = np.tile(group_labels.sum(axis=0) / n_folds, (n_folds, 1))

    group_fold = np.full(n_groups, -1, dtype=np.int64)
    unassigned = np.ones(n_groups, dtype=bool)
    remaining_label_counts = group_labels.sum(axis=0)

    def assign(g, f):
        group_fold[g] = f
        unassigned[g] = False
        demand_label[f] -= group_labels[g]
        demand_total[f] -= group_sizes[g]
        remaining_label_counts[:] -= group_labels[g]

    # distribute labeled groups, rarest remaining label first
    while True:
        active = remaining_label_counts > 0.5

        if not active.any():
            break

        counts = np.where(active, remaining_label_counts, np.inf)
        rarest = np.flatnonzero(counts == counts.min())
        label = int(rng.choice(rarest))

        cand = np.flatnonzero(unassigned & (group_labels[:, label] > 0))
        rng.shuffle(cand)

        for g in cand:
            f = _choose_fold(demand_label[:, label], demand_total, rng)
            assign(g, f)

    # groups without any positive label only need to balance fold sizes
    rest = np.flatnonzero(unassigned)
    rng.shuffle(rest)

    for g in rest:
        f = _choose_fold(demand_total, demand_total, rng)
        assign(g, f)

    assert (group_fold >= 0).all()

    # relabel folds randomly so that "the first k folds" carries no systematic bias
    perm = rng.permutation(n_folds)
    return perm[group_fold[inv]]


def nested_stratified_subsets(y, groups, fractions, n_folds=100, seed=0):
    """
    returns:
        subsets: {fraction: sorted array of record positions}
        folds:   (n_records,) small-fold id per record
    """

    folds = iterative_stratified_folds(y, groups, n_folds=n_folds, seed=seed)
    subsets = {}

    for frac in sorted(fractions):
        if frac >= 1.0:
            subsets[frac] = np.arange(len(y))
            continue

        k = int(round(frac * n_folds))

        if k < 1 or abs(k / n_folds - frac) > 1e-9:
            raise ValueError(
                f"fraction {frac} is not a multiple of 1/{n_folds}; "
                f"choose n_folds so every fraction is a whole number of folds"
            )

        subsets[frac] = np.flatnonzero(folds < k)

    # nesting check
    ordered = sorted(subsets)
    for small, large in zip(ordered[:-1], ordered[1:]):
        assert np.isin(subsets[small], subsets[large]).all(), "subsets are not nested"

    return subsets, folds


def frac_tag(frac):
    return f"{int(round(frac * 100)):03d}pct"


def label_statistics(y_train, groups_train, subsets, y_val, y_test, label_names):
    """
    returns:
        per_label: DataFrame, positive counts per label for every subset, val and test
        summary:   dict, per-fraction summary used for reporting
    """

    per_label = pd.DataFrame({"label": label_names})
    prevalence_full = y_train.mean(axis=0)
    summary = {}

    for frac in sorted(subsets):
        idx = subsets[frac]
        pos = y_train[idx].sum(axis=0).astype(int)
        per_label[f"train_pos_{frac_tag(frac)}"] = pos

        prevalence = y_train[idx].mean(axis=0)
        dev_pp = np.abs(prevalence - prevalence_full) * 100.0

        summary[frac_tag(frac)] = {
            "n_records": int(len(idx)),
            "n_patients": int(len(np.unique(groups_train[idx]))),
            "n_labels_zero_pos": int((pos == 0).sum()),
            "n_labels_1to4_pos": int(((pos >= 1) & (pos <= 4)).sum()),
            "min_pos": int(pos.min()),
            "median_pos": float(np.median(pos)),
            "n_records_without_label": int((y_train[idx].sum(axis=1) == 0).sum()),
            "mean_abs_prevalence_dev_pp": float(dev_pp.mean()),
            "max_abs_prevalence_dev_pp": float(dev_pp.max()),
        }

    per_label["val_pos"] = y_val.sum(axis=0).astype(int)
    per_label["test_pos"] = y_test.sum(axis=0).astype(int)

    summary["val"] = {
        "n_records": int(len(y_val)),
        "n_labels_zero_pos": int((y_val.sum(axis=0) == 0).sum()),
    }
    summary["test"] = {
        "n_records": int(len(y_test)),
        "n_labels_zero_pos": int((y_test.sum(axis=0) == 0).sum()),
    }

    return per_label, summary


def print_label_summary(summary):
    print("label-fraction subsets (patient-grouped, multilabel-stratified, nested):")

    for key, s in summary.items():
        if key in ("val", "test"):
            print(f"  {key:>6s}: {s['n_records']} records, {s['n_labels_zero_pos']} labels without positives")
            continue

        print(
            f"  {key:>6s}: {s['n_records']:5d} records, {s['n_patients']:5d} patients, "
            f"labels with 0 pos: {s['n_labels_zero_pos']:2d}, 1-4 pos: {s['n_labels_1to4_pos']:2d}, "
            f"median pos: {s['median_pos']:.0f}, "
            f"prevalence dev (mean/max, pp): {s['mean_abs_prevalence_dev_pp']:.3f}/{s['max_abs_prevalence_dev_pp']:.3f}"
        )


def save_label_statistics(out_dir, per_label, summary):
    out_dir = Path(out_dir)
    pid = os.getpid()

    tmp = out_dir / f".tmp{pid}_label_counts.csv"
    per_label.to_csv(tmp, index=False)
    os.replace(tmp, out_dir / "label_counts.csv")

    tmp = out_dir / f".tmp{pid}_label_summary.json"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, out_dir / "label_summary.json")
