"""
FloodSense AI - Model Evaluation Module
Evaluates a trained model checkpoint on any specified split (validation, test, or Bolivia held-out),
strictly excluding invalid pixels (LabelHand == -1 and NaN SAR).
Calculates and reports aggregated pixel-level True Positives (TP), False Positives (FP),
False Negatives (FN), True Negatives (TN), Precision, Recall, F1 Score, and IoU (Jaccard Index).
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure workspace root in path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models.dataset import Sen1Floods11Dataset
from models.unet import get_model
from models.losses import CombinedMaskedLoss
from models.metrics import evaluate_model, PixelMetrics


def run_evaluation(
    checkpoint_path: Path,
    data_dir: Path,
    split_csv: str = "flood_valid_data.csv",
    batch_size: int = 8,
    decision_threshold: float = 0.5,
    num_workers: int = 2,
    device_name: str = "auto",
    report_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluates a saved checkpoint on a target dataset split.
    """
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hyperparams = checkpoint.get("hyperparameters", {"base_channels": 32, "dropout": 0.2})
    norm_stats = checkpoint.get("norm_stats")

    if norm_stats is None:
        raise ValueError("Checkpoint does not contain normalization statistics!")

    print("\n" + "=" * 65)
    print(" FLOODSENSE AI - STANDALONE MODEL EVALUATION")
    print("=" * 65)
    print(f" Checkpoint           : {checkpoint_path.resolve()}")
    print(f" Evaluation Split     : {split_csv}")
    print(f" Compute Device       : {device}")
    print(f" Decision Threshold   : {decision_threshold}")
    print(f" Base Channels        : {hyperparams.get('base_channels', 32)}")

    # Instantiate model and load weights
    model = get_model(
        in_channels=2,
        out_channels=1,
        base_channels=hyperparams.get("base_channels", 32),
        dropout=hyperparams.get("dropout", 0.2),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create dataset using checkpoint's training normalization statistics
    eval_dataset = Sen1Floods11Dataset(
        data_dir=data_dir,
        split_csv=split_csv,
        norm_stats=norm_stats,
        augment=False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = CombinedMaskedLoss(bce_weight=0.5, dice_weight=0.5)

    print(f" Evaluating {len(eval_dataset)} chips ({len(eval_loader)} batches)...")
    avg_loss, results = evaluate_model(
        model=model,
        dataloader=eval_loader,
        criterion=criterion,
        device=device,
        threshold=decision_threshold,
    )

    tp, fp, fn, tn = results["tp"], results["fp"], results["fn"], results["tn"]

    print("\n" + "=" * 65)
    print(" EVALUATION METRICS REPORT")
    print("=" * 65)
    print(f" Average Loss (BCE+Dice) : {avg_loss:.4f}")
    print(f" Mean IoU (Jaccard Index): {results['iou']:.4f}")
    print(f" F1 Score (Dice Score)   : {results['f1']:.4f}")
    print(f" Precision               : {results['precision']:.4f}")
    print(f" Recall                  : {results['recall']:.4f}")
    print(f" Accuracy                : {results['accuracy']:.4f}")
    print(f" Total Valid Pixels      : {results['total_valid_pixels']:,} ({results['valid_fraction']*100:.2f}% of total)")
    print("\n Pixel-Level Confusion Matrix:")
    print(f"   True Positives  (TP) : {tp:>10,}")
    print(f"   False Positives (FP) : {fp:>10,}")
    print(f"   False Negatives (FN) : {fn:>10,}")
    print(f"   True Negatives  (TN) : {tn:>10,}")

    report = {
        "checkpoint": str(checkpoint_path),
        "split_csv": split_csv,
        "sample_count": len(eval_dataset),
        "decision_threshold": decision_threshold,
        "loss": avg_loss,
        "metrics": results,
    }

    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        with open(report_output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n Evaluation report saved to: {report_output.resolve()}\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate FloodSense AI Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/unet_baseline_best.pt", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--split", type=str, default="flood_valid_data.csv", help="Split CSV to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--workers", type=int, default=2, help="Worker count")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON evaluation report")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    checkpoint_p = (base_dir / args.checkpoint).resolve()
    data_dir_p = (base_dir / args.data_dir).resolve()
    output_p = (base_dir / args.output).resolve() if args.output else None

    run_evaluation(
        checkpoint_path=checkpoint_p,
        data_dir=data_dir_p,
        split_csv=args.split,
        batch_size=args.batch_size,
        decision_threshold=args.threshold,
        num_workers=args.workers,
        device_name=args.device,
        report_output=output_p,
    )


if __name__ == "__main__":
    main()
