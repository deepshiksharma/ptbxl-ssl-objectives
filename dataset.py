import ast, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb


class PTBXLCropDataset(Dataset):
    """
    input shape:    (num_records, 1000, 12)
    output shape:   (12, 250)

    mode="supervised":
        - returns: x, y, record_idx

    mode="ssl", ssl_method="mask":
        - returns: x_masked, x_clean, loss_mask, record_idx

    mode="ssl", ssl_method="denoise":
        - returns: x_noisy, x_clean, loss_mask, record_idx

    mode="ssl", ssl_method="contrastive":
        - returns: view1, view2, record_idx
    """

    def __init__(
        self,
        x, y=None, input_size=250, random_crop=True, chunkify=False, stride=125,
        mode="supervised", ssl_method=None,

        # mask params
        mask_ratio=0.5,
        mask_patch_size=25,
        mask_value=0.0,

        # denoise params
        denoise_noise_std=0.075,
        denoise_lead_dropout_prob=0.0,

        # contrastive augmentation params
        contrastive_noise_std=0.05,
        contrastive_scale_min=0.8,
        contrastive_scale_max=1.2,
        contrastive_shift_max=25,
        contrastive_time_mask_ratio=0.2,
        contrastive_time_mask_patch_size=25,
        contrastive_lead_dropout_prob=0.15,
    ):
        self.x = x
        self.y = y

        self.input_size = input_size
        self.random_crop = random_crop
        self.chunkify = chunkify
        self.stride = stride

        self.mode = mode
        self.ssl_method = ssl_method

        self.mask_ratio = mask_ratio
        self.mask_patch_size = mask_patch_size
        self.mask_value = mask_value

        self.denoise_noise_std = denoise_noise_std
        self.denoise_lead_dropout_prob = denoise_lead_dropout_prob

        self.contrastive_noise_std = contrastive_noise_std
        self.contrastive_scale_min = contrastive_scale_min
        self.contrastive_scale_max = contrastive_scale_max
        self.contrastive_shift_max = contrastive_shift_max
        self.contrastive_time_mask_ratio = contrastive_time_mask_ratio
        self.contrastive_time_mask_patch_size = contrastive_time_mask_patch_size
        self.contrastive_lead_dropout_prob = contrastive_lead_dropout_prob

        if self.mode == "supervised":
            assert self.y is not None

        if self.mode == "ssl":
            assert self.ssl_method in ["mask", "denoise", "contrastive"]

        self.index = []

        for record_idx in range(len(x)):
            record_len = x[record_idx].shape[0]

            if not chunkify:
                self.index.append((record_idx, 0, record_len))
            else:
                for start in range(0, record_len, stride):
                    end = start + input_size
                    if end <= record_len:
                        self.index.append((record_idx, start, end))

    def __len__(self):
        return len(self.index)

    def _crop_signal(self, idx):
        record_idx, start, end = self.index[idx]
        signal = self.x[record_idx]

        window_len = end - start

        if self.random_crop:
            if window_len == self.input_size:
                crop_start = start
            else:
                max_offset = window_len - self.input_size
                crop_start = start + random.randint(0, max_offset)
        else:
            crop_start = start + (window_len - self.input_size) // 2

        crop_end = crop_start + self.input_size

        crop = signal[crop_start:crop_end]          # (time, channels)
        crop = crop.T.astype(np.float32)            # (channels, time)

        return crop, record_idx

    def _apply_temporal_block_mask(self, crop):
        """
        crop: (12, 250)

        Returns:
            masked_crop: (12, 250)
            clean_crop:  (12, 250)
            loss_mask:   (1, 250), 1 where reconstruction loss is computed
        """

        clean_crop = crop.copy()
        masked_crop = crop.copy()

        n_time = crop.shape[1]
        patch = self.mask_patch_size

        assert n_time % patch == 0, "input_size must be divisible by mask_patch_size"

        n_patches = n_time // patch
        n_mask = max(1, int(round(self.mask_ratio * n_patches)))

        patch_ids = list(range(n_patches))
        masked_patch_ids = random.sample(patch_ids, n_mask)

        loss_mask = np.zeros((1, n_time), dtype=np.float32)

        for patch_id in masked_patch_ids:
            start = patch_id * patch
            end = start + patch

            loss_mask[:, start:end] = 1.0
            masked_crop[:, start:end] = self.mask_value

        return masked_crop, clean_crop, loss_mask

    def _apply_denoising_corruption(self, crop):
        """
        crop: (12, 250)

        Returns:
            noisy_crop:  (12, 250)
            clean_crop:  (12, 250)
            loss_mask:   (1, 250), all ones

        Denoising corruption:
            1. Add Gaussian noise to every lead/timepoint.
            2. Optionally drop entire leads by setting them to zero.
        """

        clean_crop = crop.copy()
        noisy_crop = crop.copy()

        if self.denoise_noise_std > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.denoise_noise_std,
                size=noisy_crop.shape,
            ).astype(np.float32)

            noisy_crop = noisy_crop + noise

        if self.denoise_lead_dropout_prob > 0:
            for lead_idx in range(noisy_crop.shape[0]):
                if random.random() < self.denoise_lead_dropout_prob:
                    noisy_crop[lead_idx, :] = 0.0

        loss_mask = np.ones((1, crop.shape[1]), dtype=np.float32)

        return noisy_crop, clean_crop, loss_mask

    def _random_amplitude_scale(self, crop):
        """
        Randomly scales the whole crop amplitude.

        crop: (12, 250)
        """

        if self.contrastive_scale_min is None or self.contrastive_scale_max is None:
            return crop

        scale = random.uniform(
            self.contrastive_scale_min,
            self.contrastive_scale_max,
        )

        return crop * scale

    def _random_gaussian_noise(self, crop):
        """
        Adds Gaussian noise to all leads/timepoints.

        crop: (12, 250)
        """

        if self.contrastive_noise_std <= 0:
            return crop

        noise = np.random.normal(
            loc=0.0,
            scale=self.contrastive_noise_std,
            size=crop.shape,
        ).astype(np.float32)

        return crop + noise

    def _random_time_shift(self, crop):
        """
        Randomly shifts the crop in time with zero fill.

        crop: (12, 250)
        """

        max_shift = self.contrastive_shift_max

        if max_shift <= 0:
            return crop

        shift = random.randint(-max_shift, max_shift)

        if shift == 0:
            return crop

        shifted = np.zeros_like(crop)

        if shift > 0:
            shifted[:, shift:] = crop[:, :-shift]
        else:
            shifted[:, :shift] = crop[:, -shift:]

        return shifted

    def _random_time_mask(self, crop):
        """
        Randomly zeros temporal patches.

        crop: (12, 250)
        """

        ratio = self.contrastive_time_mask_ratio
        patch = self.contrastive_time_mask_patch_size

        if ratio <= 0:
            return crop

        n_time = crop.shape[1]

        assert n_time % patch == 0, "input_size must be divisible by contrastive_time_mask_patch_size"

        n_patches = n_time // patch
        n_mask = int(round(ratio * n_patches))

        if n_mask <= 0:
            return crop

        n_mask = min(n_mask, n_patches)

        patch_ids = list(range(n_patches))
        masked_patch_ids = random.sample(patch_ids, n_mask)

        crop = crop.copy()

        for patch_id in masked_patch_ids:
            start = patch_id * patch
            end = start + patch
            crop[:, start:end] = 0.0

        return crop

    def _random_lead_dropout(self, crop):
        """
        Randomly zeros entire leads.

        crop: (12, 250)
        """

        p = self.contrastive_lead_dropout_prob

        if p <= 0:
            return crop

        crop = crop.copy()

        for lead_idx in range(crop.shape[0]):
            if random.random() < p:
                crop[lead_idx, :] = 0.0

        return crop

    def _apply_contrastive_augmentations(self, crop):
        """
        Creates one augmented ECG view.

        crop: (12, 250)

        Returns:
            view: (12, 250)
        """

        view = crop.copy().astype(np.float32)

        # Keep this order fixed for reproducibility/interpretability.
        view = self._random_amplitude_scale(view)
        view = self._random_time_shift(view)
        view = self._random_gaussian_noise(view)
        view = self._random_time_mask(view)
        view = self._random_lead_dropout(view)

        return view.astype(np.float32)

    def __getitem__(self, idx):
        crop, record_idx = self._crop_signal(idx)

        if self.mode == "supervised":
            label = self.y[record_idx].astype(np.float32)

            return (
                torch.from_numpy(crop),
                torch.from_numpy(label),
                record_idx,
            )

        if self.mode == "ssl":
            if self.ssl_method == "mask":
                x_corrupt, x_clean, loss_mask = self._apply_temporal_block_mask(crop)

                return (
                    torch.from_numpy(x_corrupt.astype(np.float32)),
                    torch.from_numpy(x_clean.astype(np.float32)),
                    torch.from_numpy(loss_mask.astype(np.float32)),
                    record_idx,
                )

            if self.ssl_method == "denoise":
                x_corrupt, x_clean, loss_mask = self._apply_denoising_corruption(crop)

                return (
                    torch.from_numpy(x_corrupt.astype(np.float32)),
                    torch.from_numpy(x_clean.astype(np.float32)),
                    torch.from_numpy(loss_mask.astype(np.float32)),
                    record_idx,
                )

            if self.ssl_method == "contrastive":
                view1 = self._apply_contrastive_augmentations(crop)
                view2 = self._apply_contrastive_augmentations(crop)

                return (
                    torch.from_numpy(view1.astype(np.float32)),
                    torch.from_numpy(view2.astype(np.float32)),
                    record_idx,
                )

            raise ValueError(f"Unknown ssl_method: {self.ssl_method}")

        raise RuntimeError("Invalid dataset mode")


def load_ptbxl_raw100(data_dir, cache_dir="/kaggle/working/ptbxl_cache"):
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / "raw100.npy"

    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)

    if cache_path.exists():
        x = np.load(cache_path)
        return x, db

    records = []

    for fname in db["filename_lr"]:
        signal, _ = wfdb.rdsamp(str(data_dir / fname))
        records.append(signal.astype(np.float32))

    x = np.stack(records).astype(np.float32)
    np.save(cache_path, x)

    return x, db


def get_labels(data_dir, db, task="diagnostic"):
    data_dir = Path(data_dir)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)

    if task == "all":
        label_names = scp.index.tolist()

    elif task == "diagnostic":
        label_names = scp.index[scp["diagnostic"] == 1].tolist()

    elif task == "form":
        label_names = scp.index[scp["form"] == 1].tolist()

    elif task == "rhythm":
        label_names = scp.index[scp["rhythm"] == 1].tolist()

    elif task == "superdiagnostic":
        label_names = sorted(
            scp.loc[scp["diagnostic"] == 1, "diagnostic_class"]
            .dropna()
            .unique()
            .tolist()
        )

    elif task == "subdiagnostic":
        label_names = sorted(
            scp.loc[scp["diagnostic"] == 1, "diagnostic_subclass"]
            .dropna()
            .unique()
            .tolist()
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    label_to_idx = {label: i for i, label in enumerate(label_names)}
    y = np.zeros((len(db), len(label_names)), dtype=np.float32)

    for row_idx, codes in enumerate(db["scp_codes"]):
        active_labels = set()

        for code in codes.keys():
            if code not in scp.index:
                continue

            if task in ["all", "diagnostic", "form", "rhythm"]:
                if code in label_to_idx:
                    active_labels.add(code)

            elif task == "superdiagnostic":
                if scp.loc[code].get("diagnostic", 0) == 1:
                    label = scp.loc[code, "diagnostic_class"]
                    if pd.notna(label):
                        active_labels.add(label)

            elif task == "subdiagnostic":
                if scp.loc[code].get("diagnostic", 0) == 1:
                    label = scp.loc[code, "diagnostic_subclass"]
                    if pd.notna(label):
                        active_labels.add(label)

        for label in active_labels:
            if label in label_to_idx:
                y[row_idx, label_to_idx[label]] = 1.0

    return y, label_names


def split_by_fold(x, y, db):
    folds = db["strat_fold"].values

    train_idx = np.where(folds <= 8)[0]
    val_idx = np.where(folds == 9)[0]
    test_idx = np.where(folds == 10)[0]

    return (
        x[train_idx], y[train_idx],
        x[val_idx], y[val_idx],
        x[test_idx], y[test_idx],
    )
