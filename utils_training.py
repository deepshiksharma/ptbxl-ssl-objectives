import copy, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset import PTBXLCropDataset
from model import xresnet1d101
from utils import make_loader, load_encoder_weights, safe_macro_auc


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
    pin_memory,
):
    model.eval()

    ds = PTBXLCropDataset(
        x_split,
        y_split,
        input_size=input_size,
        random_crop=False,
        chunkify=True,
        stride=stride,
        mode="supervised",
    )

    loader = make_loader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
        "preds": preds,
    }


def set_head_only_train_mode(model):
    model.eval()
    model.head.train()


def train_downstream(
    finetune_mode,
    ssl_method,
    seed,
    pretrained_ckpt,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    num_classes,
    out_dir,
    fraction,
    input_size,
    stride,
    epochs,
    batch_size,
    lr,
    weight_decay,
    device,
    num_workers,
    pin_memory,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = xresnet1d101(num_classes=num_classes, input_channels=12)
    model = load_encoder_weights(model, pretrained_ckpt)
    model = model.to(device)

    if finetune_mode == "head":
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("head.")

    train_ds = PTBXLCropDataset(
        x_train,
        y_train,
        input_size=input_size,
        random_crop=True,
        chunkify=False,
        mode="supervised",
    )

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    criterion = nn.BCEWithLogitsLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
    )

    best_val_auc = -np.inf
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        if finetune_mode == "head":
            set_head_only_train_mode(model)
        else:
            model.train()

        total_loss = 0.0
        n_batches = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

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
            input_size=input_size,
            stride=stride,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
            f"[{ssl_method} {finetune_mode} frac={fraction:.2f}] "
            f"epoch {epoch}/{epochs} "
            f"loss={train_loss:.5f} "
            f"val_auc={val_auc:.5f} "
            f"best={best_val_auc:.5f}"
        )

    model.load_state_dict(best_state)

    val_result = predict_chunked(
        model=model,
        x_split=x_val,
        y_split=y_val,
        input_size=input_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_result = predict_chunked(
        model=model,
        x_split=x_test,
        y_split=y_test,
        input_size=input_size,
        stride=stride,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    metrics = {
        "ssl_method": ssl_method,
        "seed": seed,
        "fraction": fraction,
        "finetune_mode": finetune_mode,
        "n_train_records": int(len(x_train)),
        "best_val_macro_auc": float(best_val_auc),
        "final_val_macro_auc": float(val_result["macro_auc"]),
        "test_macro_auc": float(test_result["macro_auc"]),
        "pretrained_ckpt": str(pretrained_ckpt),
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

    return metrics
