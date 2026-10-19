"""
FloodSense AI - Evaluation Metrics
Computes aggregated pixel-level True Positive (TP), False Positive (FP),
False Negative (FN), True Negative (TN), Precision, Recall, F1 Score, and IoU (Jaccard Index)
strictly excluding invalid pixels (LabelHand == -1 or NaN SAR).
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class PixelMetrics:
    """
    Accumulator for binary segmentation evaluation across multiple batches.
    Predictions are binarized using a decision threshold (default: 0.5 probability / 0.0 logit).
    Only valid pixels (valid_mask == True) are accumulated.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0
        self.total_valid_pixels: int = 0
        self.total_pixels: int = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.total_valid_pixels = 0
        self.total_pixels = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor):
        """
        Args:
            logits: Predicted logits (B, 1, H, W)
            targets: Ground-truth binary targets (B, 1, H, W) in {0.0, 1.0}
            valid_mask: Boolean tensor (B, 1, H, W), True for valid pixels
        """
        probs = torch.sigmoid(logits)
        preds = probs >= self.threshold
        trues = targets >= 0.5
        val = valid_mask.bool()

        tp = (preds & trues & val).sum().item()
        fp = (preds & (~trues) & val).sum().item()
        fn = ((~preds) & trues & val).sum().item()
        tn = ((~preds) & (~trues) & val).sum().item()

        self.tp += int(tp)
        self.fp += int(fp)
        self.fn += int(fn)
        self.tn += int(tn)
        self.total_valid_pixels += int(val.sum().item())
        self.total_pixels += int(targets.numel())

    def compute(self) -> Dict[str, Any]:
        """
        Returns aggregated confusion matrix counts and macro/micro metrics.
        """
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

        return {
            "threshold": self.threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "accuracy": float(accuracy),
            "total_valid_pixels": self.total_valid_pixels,
            "total_pixels": self.total_pixels,
            "valid_fraction": (self.total_valid_pixels / max(self.total_pixels, 1)),
        }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates a model over a DataLoader and computes average loss and pixel metrics.
    """
    model.eval()
    metrics = PixelMetrics(threshold=threshold)
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        logits = model(images)
        loss = criterion(logits, targets, valid_mask)

        total_loss += loss.item()
        num_batches += 1
        metrics.update(logits, targets, valid_mask)

    avg_loss = total_loss / max(num_batches, 1)
    results = metrics.compute()
    results["loss"] = avg_loss
    return avg_loss, results
