import sys
from pathlib import Path
import torch
from utils import env_override, parse_cli, set_seed
from utils_training import FinetuneConfig, prepare_task, run_downstream_grid


USAGE = """Usage:
    python train_baseline.py <data-dir> <task> <seed-value>
        data-dir:   PTB-XL 100 Hz directory (contains ptbxl_database.csv, scp_statements.csv, records100/)
        task:       diagnostic | superdiagnostic | subdiagnostic | form | rhythm | all
        seed-value: int

optional environment overrides (to split the grid across sessions):
    FRACTIONS=0.01,0.05   ENCODER_LRS=1e-3,1e-2   SUBSET_SEED=54
"""

DATA_DIR, TASK, _, SEED = parse_cli(sys.argv, USAGE)

METHOD = "baseline"

FS = 100
INPUT_SECONDS = 2.5
INPUT_SIZE = int(FS * INPUT_SECONDS)
STRIDE = INPUT_SIZE // 2

KERNEL_SIZE = 5
PS_HEAD = 0.5
LIN_FTRS_HEAD = (128,)

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

# the scratch baseline gets the same encoder-LR grid as the pretrained encoders; it has no
# frozen or partial counterpart because a randomly initialized encoder carries no representation
PROTOCOLS = ["full"]
ENCODER_LRS = env_override("ENCODER_LRS", [1e-4, 1e-3, 1e-2], float)
LABEL_FRACTIONS = env_override("FRACTIONS", [0.01, 0.05, 0.10, 0.25, 1.00], float)
SUBSET_SEED = env_override("SUBSET_SEED", SEED, int)
N_STRAT_FOLDS = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

set_seed(SEED)

OUT_DIR = Path(f"{METHOD}_seed{SEED}")
TASK_DIR = OUT_DIR / (TASK if SUBSET_SEED == SEED else f"{TASK}_subset{SUBSET_SEED}")

print("method:", METHOD)
print("task:", TASK)
print("seed:", SEED, "| subset seed:", SUBSET_SEED)
print("device:", DEVICE)
print("out_dir:", TASK_DIR)

data = prepare_task(DATA_DIR, TASK, SUBSET_SEED, LABEL_FRACTIONS, N_STRAT_FOLDS, TASK_DIR)

run_downstream_grid(
    method=METHOD,
    task=TASK,
    seed=SEED,
    subset_seed=SUBSET_SEED,
    pretrained_ckpt=None,
    protocols=PROTOCOLS,
    encoder_lrs=ENCODER_LRS,
    fractions=LABEL_FRACTIONS,
    data=data,
    task_dir=TASK_DIR,
    cfg=FT,
    device=DEVICE,
    extra_config={"data_dir": DATA_DIR, "n_strat_folds": N_STRAT_FOLDS},
)

print("done")
