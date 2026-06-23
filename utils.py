import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


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


def safe_macro_auc(y_true, y_prob):
    per_class_auc = []

    for c in range(y_true.shape[1]):
        yt = y_true[:, c]
        yp = y_prob[:, c]

        if yt.min() == yt.max():
            per_class_auc.append(np.nan)
        else:
            per_class_auc.append(roc_auc_score(yt, yp))

    per_class_auc = np.array(per_class_auc, dtype=np.float32)
    macro_auc = float(np.nanmean(per_class_auc))

    return macro_auc, per_class_auc


def load_encoder_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "encoder" not in ckpt:
        raise KeyError(f"checkpoint does not contain 'encoder': {ckpt_path}")

    missing, unexpected = model.load_state_dict(ckpt["encoder"], strict=False)

    print("loaded encoder:", ckpt_path)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    return model
