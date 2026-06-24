import sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import PTBXLCropDataset, load_ptbxl_raw100, get_labels, split_by_fold
from model import ReconstructionAutoencoder
from loss import reconstruction_loss
from utils_training import train_downstream
from utils import set_seed


if len(sys.argv) != 3:
    raise ValueError(
        """Usage:
        python train_reconstruction.py <recon-ssl-type> <seed-value>
            recon-ssl-type: mask | denoise
            seed-value:     int
        """
    )

SSL_METHOD = sys.argv[1]
SEED = int(sys.argv[2])

if SSL_METHOD not in ["mask", "denoise"]:
    raise ValueError("SSL_METHOD must be 'mask' or 'denoise'")


DATA_DIR = "/kaggle/input/datasets/deepshiksharma/ptb-xl-100hz/ptb-xl_100hz"

TASK = "diagnostic"

RUN_PRETRAIN = True
RUN_FULL_FINETUNE = True
RUN_HEAD_ONLY_FINETUNE = True

FS = 100
INPUT_SECONDS = 2.5
INPUT_SIZE = int(FS * INPUT_SECONDS)
STRIDE = INPUT_SIZE // 2

SSL_EPOCHS = 100
SSL_BATCH_SIZE = 128
SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-4

FINETUNE_EPOCHS = 50
FINETUNE_BATCH_SIZE = 128
FULL_FINETUNE_LR = 1e-3
HEAD_ONLY_LR = 1e-2
FINETUNE_WEIGHT_DECAY = 1e-2

LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]

KERNEL_SIZE = 5
PS_HEAD = 0.5
LIN_FTRS_HEAD = (128,)

NUM_WORKERS = 2
PIN_MEMORY = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


set_seed(SEED)

OUT_DIR = f"{SSL_METHOD}_seed{SEED}"
PRETRAIN_DIR = OUT_DIR / "pretrain"
FULL_DIR = OUT_DIR / "finetune_full"
HEAD_DIR = OUT_DIR / "finetune_head"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_CKPT = PRETRAIN_DIR / "pretrain_last.pt"
PRETRAIN_HISTORY = PRETRAIN_DIR / "pretrain_history.csv"
CONFIG_PATH = OUT_DIR / "config.json"

config = {
    "ssl_method": SSL_METHOD,
    "seed": SEED,
    "data_dir": DATA_DIR,
    "task": TASK,
    "input_size": INPUT_SIZE,
    "stride": STRIDE,
    "ssl_epochs": SSL_EPOCHS,
    "ssl_batch_size": SSL_BATCH_SIZE,
    "ssl_lr": SSL_LR,
    "ssl_weight_decay": SSL_WEIGHT_DECAY,
    "finetune_epochs": FINETUNE_EPOCHS,
    "finetune_batch_size": FINETUNE_BATCH_SIZE,
    "full_finetune_lr": FULL_FINETUNE_LR,
    "head_only_lr": HEAD_ONLY_LR,
    "finetune_weight_decay": FINETUNE_WEIGHT_DECAY,
    "label_fractions": LABEL_FRACTIONS,
    "device": str(DEVICE)
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("ssl_method:", SSL_METHOD)
print("seed:", SEED)
print("device:", DEVICE)
print("out_dir:", OUT_DIR)


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


if RUN_PRETRAIN:
    ssl_train_ds = PTBXLCropDataset(
        x_train_full,
        y=None,
        input_size=INPUT_SIZE,
        random_crop=True,
        chunkify=False,
        mode="ssl",
        ssl_method=SSL_METHOD,
    )

    ssl_train_loader = DataLoader(
        ssl_train_ds,
        batch_size=SSL_BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    model = ReconstructionAutoencoder(
        input_channels=12,
        input_size=INPUT_SIZE,
        kernel_size=KERNEL_SIZE,
        ps_head=PS_HEAD,
        lin_ftrs_head=LIN_FTRS_HEAD,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SSL_LR,
        weight_decay=SSL_WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=SSL_LR,
        epochs=SSL_EPOCHS,
        steps_per_epoch=len(ssl_train_loader),
    )

    history = []

    for epoch in range(1, SSL_EPOCHS + 1):
        model.train()

        losses = []

        for x_corrupt, x_clean, loss_mask, _ in ssl_train_loader:
            x_corrupt = x_corrupt.to(DEVICE, non_blocking=True)
            x_clean = x_clean.to(DEVICE, non_blocking=True)
            loss_mask = loss_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            recon = model(x_corrupt)
            loss = reconstruction_loss(recon, x_clean, loss_mask)

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.append(float(loss.item()))

        row = {
            "epoch": epoch,
            "ssl_method": SSL_METHOD,
            "train_loss": float(np.mean(losses)),
            "lr": float(scheduler.get_last_lr()[0]),
        }

        history.append(row)
        pd.DataFrame(history).to_csv(PRETRAIN_HISTORY, index=False)

        print(row)

        torch.save(
            {
                "encoder": model.encoder_state_dict_without_head(),
                "model": model.state_dict(),
                "epoch": epoch,
                "ssl_method": SSL_METHOD,
                "seed": SEED,
                "config": config,
                "history": history,
            },
            PRETRAIN_CKPT,
        )

    print("saved pretrain checkpoint:", PRETRAIN_CKPT)


rng = np.random.default_rng(SEED)
train_perm = rng.permutation(len(x_train_full))

all_metrics = []

for frac in LABEL_FRACTIONS:
    n_labeled = max(1, int(round(frac * len(x_train_full))))
    subset_idx = np.sort(train_perm[:n_labeled])

    frac_name = f"{int(frac * 100):03d}pct"

    x_train = x_train_full[subset_idx]
    y_train = y_train_full[subset_idx]

    if RUN_FULL_FINETUNE:
        out_dir = FULL_DIR / frac_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "train_subset_indices.npy", subset_idx)

        metrics = train_downstream(
            finetune_mode="full",
            ssl_method=SSL_METHOD,
            seed=SEED,
            pretrained_ckpt=PRETRAIN_CKPT,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            out_dir=out_dir,
            fraction=frac,
            input_size=INPUT_SIZE,
            stride=STRIDE,
            epochs=FINETUNE_EPOCHS,
            batch_size=FINETUNE_BATCH_SIZE,
            lr=FULL_FINETUNE_LR,
            weight_decay=FINETUNE_WEIGHT_DECAY,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        all_metrics.append(metrics)

    if RUN_HEAD_ONLY_FINETUNE:
        out_dir = HEAD_DIR / frac_name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "train_subset_indices.npy", subset_idx)

        metrics = train_downstream(
            finetune_mode="head",
            ssl_method=SSL_METHOD,
            seed=SEED,
            pretrained_ckpt=PRETRAIN_CKPT,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            num_classes=num_classes,
            out_dir=out_dir,
            fraction=frac,
            input_size=INPUT_SIZE,
            stride=STRIDE,
            epochs=FINETUNE_EPOCHS,
            batch_size=FINETUNE_BATCH_SIZE,
            lr=HEAD_ONLY_LR,
            weight_decay=FINETUNE_WEIGHT_DECAY,
            device=DEVICE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        all_metrics.append(metrics)

with open(OUT_DIR / "all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("done")
print(json.dumps(all_metrics, indent=2))
