"""
FloodSense AI - Masked Segmentation Losses
Provides loss functions that strictly ignore invalid pixels (LabelHand == -1 or NaN SAR)
so that invalid pixels contribute neither to the loss magnitude nor to backpropagated gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross Entropy with Logits loss evaluated only over valid pixels.
    If a batch or image contains zero valid pixels, returns a zero loss
    with gradient tracking preserved.
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits, shape (B, 1, H, W)
            targets: Binary flood targets, shape (B, 1, H, W), float in {0.0, 1.0}
            valid_mask: Boolean tensor, shape (B, 1, H, W), True for valid pixels
        """
        valid_mask_f = valid_mask.float()
        total_valid = valid_mask_f.sum()

        if total_valid < 1.0:
            # Graph preservation with zero gradient
            return logits.sum() * 0.0

        pos_wt = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype) if self.pos_weight != 1.0 else None
        per_pixel_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_wt, reduction="none"
        )
        masked_loss = (per_pixel_loss * valid_mask_f).sum() / total_valid
        return masked_loss


class MaskedDiceLoss(nn.Module):
    """
    Soft Dice Loss calculated strictly over valid pixels.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits, shape (B, 1, H, W)
            targets: Binary flood targets, shape (B, 1, H, W)
            valid_mask: Boolean tensor, shape (B, 1, H, W)
        """
        valid_mask_f = valid_mask.float()
        total_valid = valid_mask_f.sum()

        if total_valid < 1.0:
            return logits.sum() * 0.0

        probs = torch.sigmoid(logits)
        probs_v = probs * valid_mask_f
        targets_v = targets * valid_mask_f

        # Sum over spatial dimensions (H, W) per batch element
        dims = (-2, -1)
        intersection = (probs_v * targets_v).sum(dim=dims)
        cardinality = (probs_v + targets_v).sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score
        return dice_loss.mean()


class CombinedMaskedLoss(nn.Module):
    """
    Combined BCE-with-logits and Dice loss for masked flood segmentation.
    loss = bce_weight * BCE + dice_weight * Dice
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = MaskedDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets, valid_mask)
        loss_dice = self.dice(logits, targets, valid_mask)
        return self.bce_weight * loss_bce + self.dice_weight * loss_dice
