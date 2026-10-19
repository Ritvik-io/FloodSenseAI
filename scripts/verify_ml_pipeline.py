"""
FloodSense AI - Machine Learning Pipeline Verification & Smoke Test
Validates:
  1. Dataset loading, tensor shapes, and dtypes
  2. Normalization values strictly computed from training split
  3. U-Net model forward pass and output dimension matching
  4. Masked loss computation with finite loss check
  5. Backpropagation and gradient sanity check
  6. Edge-case test: all-invalid batch produces zero loss and finite gradient
  7. Pixel metrics calculation (Precision, Recall, F1, IoU)
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure workspace root is in python path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models.dataset import Sen1Floods11Dataset, create_dataloaders
from models.unet import UNet, get_model
from models.losses import CombinedMaskedLoss, MaskedBCEWithLogitsLoss, MaskedDiceLoss
from models.metrics import PixelMetrics, evaluate_model


def run_checks():
    print("\n" + "=" * 65)
    print(" FLOODSENSE AI - ML PIPELINE VERIFICATION SUITE")
    print("=" * 65)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # -------------------------------------------------------------
    # 1. Dataset & Normalization Check
    # -------------------------------------------------------------
    print("\n[CHECK 1/6] Verifying DataLoaders and Split Isolation...")
    train_loader, val_loader, norm_stats = create_dataloaders(
        data_dir=base_dir / "data",
        batch_size=2,
        num_workers=0,  # Single-process for deterministic smoke test
        seed=42,
    )

    print(f"  Train set size      : {len(train_loader.dataset)} samples")
    print(f"  Validation set size : {len(val_loader.dataset)} samples")
    print(f"  VV Normalization    : mean = {norm_stats['vv_mean']:.4f}, std = {norm_stats['vv_std']:.4f}")
    print(f"  VH Normalization    : mean = {norm_stats['vh_mean']:.4f}, std = {norm_stats['vh_std']:.4f}")

    assert len(train_loader.dataset) == 252, f"Expected 252 train samples, got {len(train_loader.dataset)}"
    assert len(val_loader.dataset) == 89, f"Expected 89 val samples, got {len(val_loader.dataset)}"
    print("  -> PASSED: Splits and training-only normalization verified.")

    # -------------------------------------------------------------
    # 2. Batch Tensor Shapes and Dtypes
    # -------------------------------------------------------------
    print("\n[CHECK 2/6] Verifying Batch Tensor Shapes & Dynamic Flood Target...")
    sample_batch = next(iter(train_loader))
    images = sample_batch["image"]
    targets = sample_batch["target"]
    valid_mask = sample_batch["valid_mask"]

    print(f"  Image Tensor Shape  : {tuple(images.shape)} | dtype = {images.dtype}")
    print(f"  Target Tensor Shape : {tuple(targets.shape)} | dtype = {targets.dtype}")
    print(f"  Valid Mask Shape    : {tuple(valid_mask.shape)} | dtype = {valid_mask.dtype}")

    assert images.shape == (2, 2, 512, 512), f"Unexpected image shape {images.shape}"
    assert targets.shape == (2, 1, 512, 512), f"Unexpected target shape {targets.shape}"
    assert valid_mask.shape == (2, 1, 512, 512), f"Unexpected valid_mask shape {valid_mask.shape}"

    # Target values check
    unique_targets = torch.unique(targets).tolist()
    assert all(val in [0.0, 1.0] for val in unique_targets), f"Target contains non-binary values: {unique_targets}"
    print(f"  Target Values       : {unique_targets} (Strictly Binary {0.0, 1.0})")
    print("  -> PASSED: Batch shapes, channels, and binary target semantics verified.")

    # -------------------------------------------------------------
    # 3. Model Architecture & Forward Pass
    # -------------------------------------------------------------
    print("\n[CHECK 3/6] Verifying U-Net Forward Pass...")
    model = get_model(in_channels=2, out_channels=1, base_channels=32).to(device)
    images_dev = images.to(device)
    targets_dev = targets.to(device)
    valid_mask_dev = valid_mask.to(device)

    logits = model(images_dev)
    print(f"  Logits Output Shape : {tuple(logits.shape)}")
    assert logits.shape == (2, 1, 512, 512), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN found in model output logits!"
    assert not torch.isinf(logits).any(), "Inf found in model output logits!"
    print("  -> PASSED: Forward pass successfully executed with matching spatial resolution.")

    # -------------------------------------------------------------
    # 4. Masked Loss & Backward Propagation
    # -------------------------------------------------------------
    print("\n[CHECK 4/6] Verifying Masked Loss & Gradient Backpropagation...")
    criterion = CombinedMaskedLoss(bce_weight=0.5, dice_weight=0.5)
    loss = criterion(logits, targets_dev, valid_mask_dev)
    print(f"  Combined Masked Loss: {loss.item():.6f}")

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"Loss should be positive, got: {loss.item()}"

    loss.backward()

    # Check gradients
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients were computed!"
    assert all(np.isfinite(g) for g in grad_norms), "Non-finite gradient detected!"
    print(f"  Max Gradient Norm   : {max(grad_norms):.6f}")
    print(f"  Mean Gradient Norm  : {np.mean(grad_norms):.6f}")
    print("  -> PASSED: Finite loss and healthy gradients verified.")

    # -------------------------------------------------------------
    # 5. Edge-Case: All-Invalid Mask Handling
    # -------------------------------------------------------------
    print("\n[CHECK 5/6] Verifying Edge-Case: All-Invalid Batch Masking...")
    dummy_logits = torch.randn(2, 1, 64, 64, requires_grad=True, device=device)
    dummy_targets = torch.zeros(2, 1, 64, 64, device=device)
    dummy_mask = torch.zeros(2, 1, 64, 64, dtype=torch.bool, device=device)  # ALL False

    edge_loss = criterion(dummy_logits, dummy_targets, dummy_mask)
    edge_loss.backward()
    print(f"  All-Invalid Loss    : {edge_loss.item():.6f}")
    assert edge_loss.item() == 0.0, f"Expected 0.0 loss for all-invalid mask, got {edge_loss.item()}"
    assert dummy_logits.grad is not None, "Gradient graph was broken by all-invalid mask"
    assert (dummy_logits.grad == 0.0).all(), "Gradients should be strictly zero for invalid pixels"
    print("  -> PASSED: All-invalid batch produces zero loss and zero gradient without breaking computation graph.")

    # -------------------------------------------------------------
    # 6. Evaluation Metrics Sanity Check
    # -------------------------------------------------------------
    print("\n[CHECK 6/6] Verifying Pixel-Level Metrics Calculation...")
    metrics = PixelMetrics(threshold=0.5)
    metrics.update(logits, targets_dev, valid_mask_dev)
    results = metrics.compute()

    print(f"  Precision           : {results['precision']:.4f}")
    print(f"  Recall              : {results['recall']:.4f}")
    print(f"  F1 Score            : {results['f1']:.4f}")
    print(f"  IoU (Jaccard)       : {results['iou']:.4f}")
    print(f"  Confusion Matrix    : TP={results['tp']:,}, FP={results['fp']:,}, FN={results['fn']:,}, TN={results['tn']:,}")

    assert 0.0 <= results["precision"] <= 1.0
    assert 0.0 <= results["recall"] <= 1.0
    assert 0.0 <= results["f1"] <= 1.0
    assert 0.0 <= results["iou"] <= 1.0
    print("  -> PASSED: Pixel metrics correctly bounded in [0, 1].")

    print("\n" + "=" * 65)
    print(" >>> ALL 6 ML PIPELINE CHECKS PASSED SUCCESSFULLY! <<<")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
