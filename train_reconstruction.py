import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from compute import RunCompute, Stopwatch, forward_flops, hardware_info
from dataset import PTBXLCropDataset
from loss import reconstruction_loss
from model import ReconstructionAutoencoder
from utils import env_override, file_sha256, parse_cli, set_seed
from utils_pretrain import pretraining_param_counts, resolve_pretrain_mode, summarize_pretrain_compute
from utils_training import FinetuneConfig, atomic_json, prepare_task, run_downstream_grid


USAGE = """Usage:
    python train_reconstruction.py <data-dir> <task> <recon-ssl-type> <seed-value>
        data-dir:       PTB-XL 100 Hz directory (contains ptbxl_database.csv, scp_statements.csv, records100/)
        task:           diagnostic | superdiagnostic | subdiagnostic | form | rhythm | all
        recon-ssl-type: mask | denoise
        seed-value:     int

optional environment overrides:
    PRETRAIN=auto|reuse|train   PROFILE_PRETRAIN_EPOCHS=3
    FRACTIONS=0.01,0.05   PROTOCOLS=frozen,partial,full   ENCODER_LRS=1e-3,1e-2   SUBSET_SEED=54
"""

DATA_DIR, TASK, (SSL_METHOD,), SEED = parse_cli(sys.argv, USAGE, middle=("recon-ssl-type",))

if SSL_METHOD not in ["mask", "denoise"]:
    raise ValueError(f"recon-ssl-type must be 'mask' or 'denoise', got {SSL_METHOD!r}\n{USAGE}")

DATASET_SSL_METHOD = SSL_METHOD

FS = 100
INPUT_SECONDS = 2.5
INPUT_SIZE = int(FS * INPUT_SECONDS)
STRIDE = INPUT_SIZE // 2

# pretraining (unchanged from the original experiments)
SSL_EPOCHS = 100
SSL_BATCH_SIZE = 128
SSL_LR = 1e-3
SSL_WEIGHT_DECAY = 1e-4

KERNEL_SIZE = 5
PS_HEAD = 0.5
LIN_FTRS_HEAD = (128,)

NUM_WORKERS = 2
PIN_MEMORY = True

PRETRAIN_MODE = env_override("PRETRAIN", "auto", str)
PROFILE_PRETRAIN_EPOCHS = env_override("PROFILE_PRETRAIN_EPOCHS", 0, int)

# downstream: identical settings for every method; only the encoder initialization differs
FT = FinetuneConfig(
    epochs=50,
    batch_size=128,
    head_lr=1e-2,
    weight_decay=1e-2,
    input_size=INPUT_SIZE,
    stride=STRIDE,
    eval_batch_size=512,
    kernel_size=KERNEL_SIZE,
    ps_head=PS_HEAD,
    lin_ftrs_head=LIN_FTRS_HEAD,
)

PROTOCOLS = env_override("PROTOCOLS", ["frozen", "partial", "full"], str)
ENCODER_LRS = env_override("ENCODER_LRS", [1e-4, 1e-3, 1e-2], float)
LABEL_FRACTIONS = env_override("FRACTIONS", [0.01, 0.05, 0.10, 0.25, 1.00], float)
SUBSET_SEED = env_override("SUBSET_SEED", SEED, int)
N_STRAT_FOLDS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

OUT_DIR = Path(f"{SSL_METHOD}_seed{SEED}")
PRETRAIN_DIR = OUT_DIR / "pretrain"
PRETRAIN_CKPT = PRETRAIN_DIR / "pretrain_last.pt"
TASK_DIR = OUT_DIR / (TASK if SUBSET_SEED == SEED else f"{TASK}_subset{SUBSET_SEED}")

pretrain_config = {
    "ssl_method": SSL_METHOD,
    "dataset_ssl_method": DATASET_SSL_METHOD,
    "seed": SEED,
    "input_size": INPUT_SIZE,
    "ssl_epochs": SSL_EPOCHS,
    "ssl_batch_size": SSL_BATCH_SIZE,
    "ssl_lr": SSL_LR,
    "ssl_weight_decay": SSL_WEIGHT_DECAY,
    "device": str(DEVICE),
}

print("ssl_method:", SSL_METHOD)
print("task:", TASK)
print("seed:", SEED, "| subset seed:", SUBSET_SEED)
print("device:", DEVICE)
print("out_dir:", OUT_DIR)


def run_pretraining(x_train_full, n_epochs_run, out_dir, save_checkpoint):
    out_dir.mkdir(parents=True, exist_ok=True)

    ssl_train_ds = PTBXLCropDataset(
        x_train_full,
        y=None,
        input_size=INPUT_SIZE,
        random_crop=True,
        chunkify=False,
        mode="ssl",
        ssl_method=DATASET_SSL_METHOD,
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
    )

    example = torch.zeros(2, 12, INPUT_SIZE)
    model_stats = {
        **pretraining_param_counts(model),
        "forward_flops_per_training_sample": forward_flops(model, example),
        "flops_note": "torch FlopCounterMode, FLOPs = 2 x MACs; one sample = one corrupted crop through encoder and decoder",
    }

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SSL_LR,
        weight_decay=SSL_WEIGHT_DECAY,
    )

    # the schedule always spans SSL_EPOCHS, also when profiling only a few epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=SSL_LR,
        epochs=SSL_EPOCHS,
        steps_per_epoch=len(ssl_train_loader),
    )

    history = []
    epoch_times = []
    rc = RunCompute().start()

    for epoch in range(1, n_epochs_run + 1):
        sw = Stopwatch().start()

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

        epoch_s = sw.stop()
        epoch_times.append(epoch_s)

        row = {
            "epoch": epoch,
            "ssl_method": SSL_METHOD,
            "train_loss": float(np.mean(losses)),
            "lr": float(scheduler.get_last_lr()[0]),
            "epoch_time_s": epoch_s,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "pretrain_history.csv", index=False)

        print(row)

        if save_checkpoint:
            torch.save(
                {
                    "encoder": model.encoder_state_dict_without_head(),
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "ssl_method": SSL_METHOD,
                    "seed": SEED,
                    "config": pretrain_config,
                    "history": history,
                },
                out_dir / "pretrain_last.pt",
            )

    compute = summarize_pretrain_compute(
        rc.finish(), epoch_times, SSL_EPOCHS, len(ssl_train_loader),
        {"model_stats": model_stats, "hardware": hardware_info(), "config": pretrain_config},
    )
    atomic_json(out_dir / "pretrain_compute.json", compute)

    print(f"pretraining: {len(epoch_times)} epochs, median {compute['epoch_time_s_median_excl_first']:.1f} s/epoch, "
          f"peak mem {compute.get('peak_memory_allocated_mb', 0):.0f} MB")


data = prepare_task(DATA_DIR, TASK, SUBSET_SEED, LABEL_FRACTIONS, N_STRAT_FOLDS, TASK_DIR)

if PROFILE_PRETRAIN_EPOCHS > 0:
    run_pretraining(data["x_train"], PROFILE_PRETRAIN_EPOCHS, OUT_DIR / "pretrain_profile", save_checkpoint=False)
    print("profiling done; no checkpoint written, no downstream runs")
    sys.exit(0)

if resolve_pretrain_mode(PRETRAIN_MODE, PRETRAIN_CKPT, SSL_EPOCHS) == "train":
    run_pretraining(data["x_train"], SSL_EPOCHS, PRETRAIN_DIR, save_checkpoint=True)
else:
    print(f"reusing pretrained encoder: {PRETRAIN_CKPT} (sha256 {file_sha256(PRETRAIN_CKPT)[:12]}...)")

run_downstream_grid(
    method=SSL_METHOD,
    task=TASK,
    seed=SEED,
    subset_seed=SUBSET_SEED,
    pretrained_ckpt=PRETRAIN_CKPT,
    protocols=PROTOCOLS,
    encoder_lrs=ENCODER_LRS,
    fractions=LABEL_FRACTIONS,
    data=data,
    task_dir=TASK_DIR,
    cfg=FT,
    device=DEVICE,
    extra_config={"data_dir": DATA_DIR, "n_strat_folds": N_STRAT_FOLDS, "pretrain_config": pretrain_config},
)

print("done")
