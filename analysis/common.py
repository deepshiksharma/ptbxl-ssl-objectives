# Constants and helpers shared by the analysis scripts

import numpy as np
import pandas as pd
import hashlib, json, re
from pathlib import Path

"""
expected layout of <runs-dir>:
    <method>_seed<seed>/
        pretrain/   # (SSL methods only)
        <task>/
            labels.npz, subsets.npz, label_counts.csv, label_summary.json, ...
            <config>/<NNN>pct/metrics.json, history.csv, compute.json, run_config.json, *_logits.npy, best_model.pt
"""


METHODS = ["baseline", "mask", "denoise", "simclr", "byol"]
SSL_METHODS = ["mask", "denoise", "simclr", "byol"]
METHOD_NAMES = {"baseline": "Scratch", "mask": "Masking", "denoise": "Denoising", "simclr": "SimCLR", "byol": "BYOL"}

PROTOCOLS = ["frozen", "partial", "full"]
PROTOCOL_NAMES = {"frozen": "Frozen", "partial": "Partial", "full": "Full"}

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
ENCODER_LRS = [1e-4, 1e-3, 1e-2]
HEAD_LR = 1e-2

N_FINETUNE_EPOCHS = 50
N_PRETRAIN_EPOCHS = 100


def frac_tag(frac):
    return f"{int(round(frac * 100)):03d}pct"


def run_name(protocol, encoder_lr, head_lr=HEAD_LR):
    if protocol == "frozen":
        return f"frozen_head{head_lr:.0e}"
    return f"{protocol}_enc{encoder_lr:.0e}_head{head_lr:.0e}"


def expected_configs(method):
    if method == "baseline":
        return [("full", lr) for lr in ENCODER_LRS]
    return [("frozen", 0.0)] + [(p, lr) for p in ("partial", "full") for lr in ENCODER_LRS]


def find_sessions(runs_dir):
    # returns sorted [(method, seed, path)] for every <method>_seed<seed> folder

    sessions = []

    for path in Path(runs_dir).iterdir():
        m = re.fullmatch(r"(\w+?)_seed(\d+)", path.name)

        if path.is_dir() and m and m.group(1) in METHODS:
            sessions.append((m.group(1), int(m.group(2)), path))

    return sorted(sessions, key=lambda s: (METHODS.index(s[0]), s[1]))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_text(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_runs(out_dir):
    # the master table written by 01_collect_runs.py

    path = Path(out_dir) / "runs.csv"

    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run 01_collect_runs.py first")

    return pd.read_csv(path)


def array_digest(arrays):
    # content hash of a dict of numpy arrays (independent of file timestamps)

    h = hashlib.sha256()

    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())

    return h.hexdigest()


def file_sha256(path, chunk=2**20):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)

    return h.hexdigest()
