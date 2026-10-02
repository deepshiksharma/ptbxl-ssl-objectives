import os, random, hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


VALID_TASKS = ["diagnostic", "superdiagnostic", "subdiagnostic", "form", "rhythm", "all"]


def parse_cli(argv, usage, middle=()):
    """
    parses: <data-dir> <task> [middle args...] <seed>

    middle: names of extra positional arguments between task and seed,
            e.g. ("recon-ssl-type",) for train_reconstruction.py
    returns: data_dir (str), task (str), middle values (list of str), seed (int)
    """

    expected = 3 + len(middle)

    if len(argv) != expected + 1:
        raise ValueError(usage)

    data_dir = argv[1]
    task = argv[2]
    middle_values = list(argv[3:3 + len(middle)])

    try:
        seed = int(argv[-1])
    except ValueError:
        raise ValueError(f"seed must be an integer, got {argv[-1]!r}\n{usage}")

    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {VALID_TASKS}, got {task!r}\n{usage}")

    if not (Path(data_dir) / "ptbxl_database.csv").exists():
        raise FileNotFoundError(f"ptbxl_database.csv not found in data dir: {data_dir}")

    return data_dir, task, middle_values, seed


def env_override(name, default, cast):
    """
    optional override through an environment variable, used to split the run grid
    across Kaggle sessions without editing files, e.g.
        FRACTIONS=0.01,0.05 PROTOCOLS=full ENCODER_LRS=1e-3 python train_contrastive_byol.py ...
    lists are comma-separated.
    """

    raw = os.environ.get(name)

    if raw is None or raw.strip() == "":
        return default

    if isinstance(default, (list, tuple)):
        return [cast(v.strip()) for v in raw.split(",") if v.strip()]

    return cast(raw)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_loader(dataset, batch_size, shuffle, drop_last, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _per_class(metric_fn, y_true, y_prob):
    """classes whose labels are all 0 (or all 1) in y_true are undefined and returned as nan"""

    out = []

    for c in range(y_true.shape[1]):
        yt = y_true[:, c]

        if yt.min() == yt.max():
            out.append(np.nan)
        else:
            out.append(metric_fn(yt, y_prob[:, c]))

    return np.array(out, dtype=np.float64)


def safe_macro_auc(y_true, y_prob):
    per_class_auc = _per_class(roc_auc_score, y_true, y_prob)
    return float(np.nanmean(per_class_auc)), per_class_auc


def safe_macro_auprc(y_true, y_prob):
    per_class_ap = _per_class(average_precision_score, y_true, y_prob)
    return float(np.nanmean(per_class_ap)), per_class_ap


def load_encoder_weights(model, ckpt_path):
    """
    Loads the pretrained encoder into a fresh xresnet1d101.
    Note: Only the (randomly initialized) classification head may be missing; anything else missing or unexpected is an error, not a warning.
    """

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "encoder" not in ckpt:
        raise KeyError(f"checkpoint does not contain 'encoder': {ckpt_path}")

    missing, unexpected = model.load_state_dict(ckpt["encoder"], strict=False)
    bad_missing = [k for k in missing if not k.startswith("head.")]

    if bad_missing or unexpected:
        raise RuntimeError(
            f"encoder checkpoint mismatch: {len(bad_missing)} non-head keys missing, "
            f"{len(unexpected)} unexpected keys. first few: {bad_missing[:3]} {list(unexpected)[:3]}"
        )

    print(f"loaded encoder: {ckpt_path} (head keys left at init: {len(missing)})")

    return model


def file_sha256(path, chunk=2**20):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)

    return h.hexdigest()
