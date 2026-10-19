"""
Unit tests for FloodSense AI ML baseline modules:
  - U-Net forward pass and shape preservation
  - Masked loss functions and edge-case invalid mask handling
  - Pixel-level evaluation metrics calculation
"""

import pytest
import torch
import torch.nn as nn

from models.unet import get_model, UNet
from models.losses import CombinedMaskedLoss, MaskedBCEWithLogitsLoss, MaskedDiceLoss
from models.metrics import PixelMetrics


def test_unet_output_shape():
    """Verifies that U-Net produces a single logit per pixel matching input spatial size."""
    model = get_model(in_channels=2, out_channels=1, base_channels=16)
    x = torch.randn(2, 2, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)
    assert not torch.isnan(out).any()


def test_masked_bce_excludes_invalid_pixels():
    """Verifies that pixels with valid_mask=False do not affect BCE loss."""
    criterion = MaskedBCEWithLogitsLoss()
    logits = torch.tensor([[[[2.0, -2.0], [5.0, -5.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    mask = torch.tensor([[[[True, True], [False, False]]]])  # Only top row is valid

    loss = criterion(logits, targets, mask)
    loss.backward()

    # Bottom row should have strictly zero gradients
    assert logits.grad[0, 0, 1, 0].item() == 0.0
    assert logits.grad[0, 0, 1, 1].item() == 0.0
    # Top row should have non-zero gradients
    assert logits.grad[0, 0, 0, 0].item() != 0.0


def test_masked_loss_all_invalid_edge_case():
    """Verifies that an all-invalid batch produces zero loss and zero gradient without crashing."""
    criterion = CombinedMaskedLoss(bce_weight=0.5, dice_weight=0.5)
    logits = torch.randn(2, 1, 32, 32, requires_grad=True)
    targets = torch.zeros(2, 1, 32, 32)
    mask = torch.zeros(2, 1, 32, 32, dtype=torch.bool)  # All invalid

    loss = criterion(logits, targets, mask)
    loss.backward()

    assert loss.item() == 0.0
    assert (logits.grad == 0.0).all()


def test_pixel_metrics_perfect_prediction():
    """Verifies that perfect predictions yield 1.0 for Precision, Recall, F1, and IoU."""
    metrics = PixelMetrics(threshold=0.5)
    logits = torch.tensor([[[[5.0, -5.0], [-5.0, 5.0]]]])   # Sigmoid ~ 1, 0, 0, 1
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    mask = torch.tensor([[[[True, True], [True, True]]]])

    metrics.update(logits, targets, mask)
    res = metrics.compute()

    assert res["tp"] == 2
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["tn"] == 2
    assert pytest.approx(res["precision"], 1e-4) == 1.0
    assert pytest.approx(res["recall"], 1e-4) == 1.0
    assert pytest.approx(res["f1"], 1e-4) == 1.0
    assert pytest.approx(res["iou"], 1e-4) == 1.0
