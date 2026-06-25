import sys, copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import PTBXLCropDataset, load_ptbxl_raw100, get_labels, split_by_fold
from model import xresnet1d101
from utils import set_seed, make_loader, safe_macro_auc


if len(sys.argv) != 2:
    raise ValueError(
        """Usage:
        python train_baseline.py <seed-value>
            seed-value: int
        """
    )

SEED = int(sys.argv[1])

METHOD = "baseline"

DATA_DIR = "ptb-xl_100hz"

TASK = "diagnostic"

RUN_BASELINE = True

FS = 100
INPUT_SECONDS = 2.5
INPUT_SIZE = int(FS * INPUT_SECONDS)
STRIDE = INPUT_SIZE // 2

EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-2
WEIGHT_DECAY = 1e-2

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]

KERNEL_SIZE = 5
PS_HEAD = 0.5
LIN_FTRS_HEAD = (128,)

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


set_seed(SEED)

OUT_DIR = Path(f"{METHOD}_seed{SEED}")
FULL_DIR = OUT_DIR / "finetune_full"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = OUT_DIR / "config.json"

config = {
    "method": METHOD,
    "seed": SEED,
    "data_dir": DATA_DIR,
    "task": TASK,
    "input_size": INPUT_SIZE,
    "stride": STRIDE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "label_fractions": LABEL_FRACTIONS,
    "device": str(DEVICE)
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("method:", METHOD)
print("seed:", SEED)
print("device:", DEVICE)
print("out_dir:", OUT_DIR)


@torch.no_grad()
def predict_chunked(
    model,
    x_split,
    y_split,
    input_size,
    stride,
    batch_size,
    device,
    num_workers,
    pin_memory
):
    model.eval()

    ds = PTBXLCropDataset(
        x_split,
        y_split,
        input_size=input_size,
        random_crop=False,
        chunkify=True,
        stride=stride,
        mode="supervised"
    )

    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    n_records = len(x_split)
    n_classes = y_split.shape[1]

    preds = np.full((n_records, n_classes), -np.inf, dtype=np.float32)

    for xb, _, record_idx in loader:
        xb = xb.to(device, non_blocking=True)

        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        record_idx = record_idx.numpy()

        for i, ridx in enumerate(record_idx):
            preds[ridx] = np.maximum(preds[ridx], probs[i])

    macro_auc, per_class_auc = safe_macro_auc(y_split, preds)

    return {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "preds": preds
    }


x, db = load_ptbxl_raw100(DATA_DIR)
y, label_names = get_labels(DATA_DIR, db, task=TASK)

x_train_full, y_train_full, x_val, y_val, x_test, y_test = split_by_fold(x, y, db)
num_classes = y_train_full.shape[1]

print("x:", x.shape)
print("y:", y.shape)
print("train:", x_train_full.shape, y_train_full.shape)
print("val:", x_val.shape, y_val.shape)
print("test:", x_test.shape, y_test.shape)
print("num_classes:", num_classes)


rng = np.random.default_rng(SEED)
train_perm = rng.permutation(len(x_train_full))

all_metrics = []

for frac in LABEL_FRACTIONS:
    n_labeled = max(1, int(round(frac * len(x_train_full))))
    subset_idx = np.sort(train_perm[:n_labeled])

    frac_name = f"{int(frac * 100):03d}pct"
    out_dir = FULL_DIR / frac_name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "train_subset_indices.npy", subset_idx)

    x_train = x_train_full[subset_idx]
    y_train = y_train_full[subset_idx]

    train_ds = PTBXLCropDataset(
        x_train,
        y_train,
        input_size=INPUT_SIZE,
        random_crop=True,
        chunkify=False,
        mode="supervised",
    )

    train_loader = make_loader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = xresnet1d101(
        num_classes=num_classes,
        input_channels=12,
        kernel_size=KERNEL_SIZE,
        ps_head=PS_HEAD,
        lin_ftrs_head=LIN_FTRS_HEAD,
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
    )

    best_val_auc = -np.inf
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        val_result = predict_chunked(
            model=model,
            x_split=x_val,
            y_split=y_val,
            input_size=INPUT_SIZE,
            stride=STRIDE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        val_auc = val_result["macro_auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_auc": val_auc,
            "best_val_macro_auc": best_val_auc,
            "lr": float(scheduler.get_last_lr()[0]),
        }

        history.append(row)

        print(
            f"[{METHOD} full frac={frac:.2f}] "
            f"epoch {epoch}/{EPOCHS} "
            f"loss={train_loss:.5f} "
            f"val_auc={val_auc:.5f} "
            f"best={best_val_auc:.5f}"
        )

    model.load_state_dict(best_state)

    val_result = predict_chunked(
        model=model,
        x_split=x_val,
        y_split=y_val,
        input_size=INPUT_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_result = predict_chunked(
        model=model,
        x_split=x_test,
        y_split=y_test,
        input_size=INPUT_SIZE,
        stride=STRIDE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    metrics = {
        "method": METHOD,
        "ssl_method": None,
        "seed": SEED,
        "fraction": frac,
        "finetune_mode": "full",
        "n_train_records": int(len(x_train)),
        "best_val_macro_auc": float(best_val_auc),
        "final_val_macro_auc": float(val_result["macro_auc"]),
        "test_macro_auc": float(test_result["macro_auc"]),
        "pretrained_ckpt": None,
    }

    torch.save(
        {
            "model": model.state_dict(),
            "metrics": metrics,
            "history": history,
        },
        out_dir / "best_model.pt",
    )

    np.save(out_dir / "val_preds.npy", val_result["preds"])
    np.save(out_dir / "test_preds.npy", test_result["preds"])
    np.save(out_dir / "val_per_class_auc.npy", val_result["per_class_auc"])
    np.save(out_dir / "test_per_class_auc.npy", test_result["per_class_auc"])

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    all_metrics.append(metrics)

with open(OUT_DIR / "all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("done")
print(json.dumps(all_metrics, indent=2))
