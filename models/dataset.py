"""
FloodSense AI - PyTorch Dataset & DataLoader for Sen1Floods11
Reads Sentinel-1 VV/VH SAR imagery, ground-truth LabelHand masks, and aligned JRCWaterHand rasters.
Constructs binary dynamic flood targets: (LabelHand == 1) & (JRCWaterHand == 0)
Masks out invalid pixels: LabelHand == -1 and SAR NaNs
Computes normalization statistics strictly from the training split.
Applies synchronized spatial augmentations across imagery, labels, and validity masks.
"""

import os
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader


def compute_training_normalization(
    data_dir: Path,
    train_split_csv: str = "flood_train_data.csv",
    cache_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Computes mean and standard deviation of VV and VH polarizations
    strictly across valid (non-NaN and LabelHand != -1) training set pixels.
    Caches the results to disk to avoid redundant recomputation.
    """
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
                if all(k in stats for k in ["vv_mean", "vv_std", "vh_mean", "vh_std"]):
                    return stats
        except Exception:
            pass

    csv_path = data_dir / "splits" / train_split_csv
    s1_dir = data_dir / "raw" / "S1Hand"
    lbl_dir = data_dir / "masks" / "LabelHand"

    if not csv_path.exists():
        raise FileNotFoundError(f"Training split CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    vv_sum = 0.0
    vh_sum = 0.0
    vv_sq_sum = 0.0
    vh_sq_sum = 0.0
    total_valid = 0

    for row in rows:
        if len(row) < 2:
            continue
        s1_name, lbl_name = row[0].strip(), row[1].strip()
        s1_p = s1_dir / s1_name
        lbl_p = lbl_dir / lbl_name
        if not s1_p.exists() or not lbl_p.exists():
            continue

        with rasterio.open(s1_p) as s1_src, rasterio.open(lbl_p) as lbl_src:
            s1 = s1_src.read()  # (2, H, W)
            lbl = lbl_src.read(1)  # (H, W)

            vv = s1[0]
            vh = s1[1]

            valid = (lbl != -1) & (~np.isnan(vv)) & (~np.isnan(vh))
            vv_valid = vv[valid]
            vh_valid = vh[valid]

            n = len(vv_valid)
            if n > 0:
                total_valid += n
                vv_sum += float(np.sum(vv_valid))
                vh_sum += float(np.sum(vh_valid))
                vv_sq_sum += float(np.sum(vv_valid ** 2))
                vh_sq_sum += float(np.sum(vh_valid ** 2))

    if total_valid == 0:
        raise ValueError("No valid training pixels found to compute normalization statistics.")

    vv_mean = vv_sum / total_valid
    vh_mean = vh_sum / total_valid
    vv_std = float(np.sqrt(max(vv_sq_sum / total_valid - vv_mean ** 2, 1e-6)))
    vh_std = float(np.sqrt(max(vh_sq_sum / total_valid - vh_mean ** 2, 1e-6)))

    stats = {
        "vv_mean": float(vv_mean),
        "vv_std": float(vv_std),
        "vh_mean": float(vh_mean),
        "vh_std": float(vh_std),
        "total_valid_pixels": int(total_valid),
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return stats


class Sen1Floods11Dataset(Dataset):
    """
    PyTorch Dataset for Sen1Floods11 Hand-Labeled flood segmentation.
    Features:
      - Sentinel-1 SAR VV and VH polarizations (2 channels).
      - Ground-truth binary dynamic-flood target: (LabelHand == 1) & (JRCWaterHand == 0).
      - Validity mask: excludes LabelHand == -1 and NaN SAR pixels.
      - Normalization using training-set statistics.
      - Synchronized spatial augmentations (random horizontal flip, vertical flip, rot90).
    """
    def __init__(
        self,
        data_dir: Path,
        split_csv: str = "flood_train_data.csv",
        norm_stats: Optional[Dict[str, float]] = None,
        augment: bool = False,
        seed: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split_csv = split_csv
        self.augment = augment
        self.seed = seed

        self.s1_dir = self.data_dir / "raw" / "S1Hand"
        self.lbl_dir = self.data_dir / "masks" / "LabelHand"
        self.jrc_dir = self.data_dir / "raw" / "JRCWaterHand"
        self.splits_dir = self.data_dir / "splits"

        csv_file = self.splits_dir / split_csv
        if not csv_file.exists():
            raise FileNotFoundError(f"Split CSV not found: {csv_file}")

        self.samples: List[Tuple[str, str, str]] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    s1_name = row[0].strip()
                    lbl_name = row[1].strip()
                    jrc_name = s1_name.replace("_S1Hand.tif", "_JRCWaterHand.tif")
                    self.samples.append((s1_name, lbl_name, jrc_name))

        if not self.samples:
            raise ValueError(f"No samples found in split CSV: {csv_file}")

        # Normalization statistics
        if norm_stats is not None:
            self.norm_stats = norm_stats
        else:
            cache_p = self.splits_dir / "train_norm_stats.json"
            self.norm_stats = compute_training_normalization(self.data_dir, cache_path=cache_p)

        self.vv_mean = self.norm_stats["vv_mean"]
        self.vv_std = self.norm_stats["vv_std"]
        self.vh_mean = self.norm_stats["vh_mean"]
        self.vh_std = self.norm_stats["vh_std"]

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentations(
        self, image: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synchronized spatial augmentations applied identically to image, target, and valid_mask.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-1])
            target = torch.flip(target, dims=[-1])
            valid_mask = torch.flip(valid_mask, dims=[-1])

        # Random vertical flip
        if random.random() > 0.5:
            image = torch.flip(image, dims=[-2])
            target = torch.flip(target, dims=[-2])
            valid_mask = torch.flip(valid_mask, dims=[-2])

        # Random 90-degree rotations (k in [0, 1, 2, 3])
        k = random.randint(0, 3)
        if k > 0:
            image = torch.rot90(image, k=k, dims=[-2, -1])
            target = torch.rot90(target, k=k, dims=[-2, -1])
            valid_mask = torch.rot90(valid_mask, k=k, dims=[-2, -1])

        return image, target, valid_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s1_name, lbl_name, jrc_name = self.samples[idx]

        s1_path = self.s1_dir / s1_name
        lbl_path = self.lbl_dir / lbl_name
        jrc_path = self.jrc_dir / jrc_name

        # Read Sentinel-1 SAR (2 bands: VV and VH)
        with rasterio.open(s1_path) as s1_src:
            s1_data = s1_src.read()  # (2, 512, 512), float32
            vv = s1_data[0].astype(np.float32)
            vh = s1_data[1].astype(np.float32)

        # Read Ground-Truth Mask (1 band)
        with rasterio.open(lbl_path) as lbl_src:
            lbl_data = lbl_src.read(1)  # (512, 512), int16 (-1, 0, 1)

        # Read Permanent Water Mask (1 band)
        if jrc_path.exists():
            with rasterio.open(jrc_path) as jrc_src:
                jrc_data = jrc_src.read(1)  # (512, 512), uint8 (0, 1)
        else:
            jrc_data = np.zeros_like(lbl_data, dtype=np.uint8)

        # Construct validity mask:
        # A pixel is valid if LabelHand != -1 and neither VV nor VH is NaN or infinite
        valid_np = (lbl_data != -1) & (~np.isnan(vv)) & (~np.isnan(vh)) & (~np.isinf(vv)) & (~np.isinf(vh))

        # Dynamic flood target: (LabelHand == 1) & (JRCWaterHand == 0)
        # Excludes permanent water bodies (JRCWaterHand == 1) from the flood target
        flood_target_np = ((lbl_data == 1) & (jrc_data == 0)).astype(np.float32)
        # Zero out invalid pixels in target (they will be masked during loss anyway)
        flood_target_np[~valid_np] = 0.0

        # Normalize SAR polarizations using training-set statistics
        # Replace NaNs with training-set mean so normalized value is 0.0 (prevents NaN conv propagation)
        vv_clean = np.where(valid_np, vv, self.vv_mean)
        vh_clean = np.where(valid_np, vh, self.vh_mean)

        vv_norm = (vv_clean - self.vv_mean) / self.vv_std
        vh_norm = (vh_clean - self.vh_mean) / self.vh_std

        # Stack SAR into (2, H, W)
        image_tensor = torch.from_numpy(np.stack([vv_norm, vh_norm], axis=0)).float()
        target_tensor = torch.from_numpy(flood_target_np[np.newaxis, ...]).float()
        valid_tensor = torch.from_numpy(valid_np[np.newaxis, ...]).bool()

        # Synchronized spatial augmentations for training
        if self.augment:
            image_tensor, target_tensor, valid_tensor = self._apply_augmentations(
                image_tensor, target_tensor, valid_tensor
            )

        return {
            "image": image_tensor,          # (2, 512, 512) float32
            "target": target_tensor,        # (1, 512, 512) float32 in {0.0, 1.0}
            "valid_mask": valid_tensor,    # (1, 512, 512) bool
            "chip_id": s1_name.replace("_S1Hand.tif", ""),
        }


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, Dict[str, float]]:
    """
    Creates reproducible training and validation DataLoaders.
    Strictly uses flood_train_data.csv for training and flood_valid_data.csv for validation.
    Calculates normalization statistics ONLY on the training split.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / "splits" / "train_norm_stats.json"

    # Compute normalization statistics strictly on the training set
    norm_stats = compute_training_normalization(data_dir, train_split_csv="flood_train_data.csv", cache_path=cache_path)

    # Deterministic seeding generator for DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        split_csv="flood_train_data.csv",
        norm_stats=norm_stats,
        augment=True,
        seed=seed,
    )

    val_dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        split_csv="flood_valid_data.csv",
        norm_stats=norm_stats,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=g,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, norm_stats
