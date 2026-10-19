"""
FloodSense AI - Model Training Pipeline
Trains the baseline U-Net on Sentinel-1 SAR imagery using the audited Sen1Floods11 dataset.
Evaluates epoch-level metrics on the validation split excluding invalid pixels.
Saves the best checkpoint based on validation IoU, along with training logs and normalization metadata.
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure workspace root is in sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

# Internal modules
from models.dataset import create_dataloaders
from models.unet import UNet, get_model
from models.losses import CombinedMaskedLoss
from models.metrics import evaluate_model, PixelMetrics


def set_seed(seed: int = 42):
    """Sets deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch.
    Returns average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets, valid_mask)

        # Backpropagation
        loss.backward()
        # Gradient clipping to stabilize training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def run_training(
    data_dir: Path,
    checkpoints_dir: Path,
    runs_dir: Path,
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 0.0005,
    weight_decay: float = 0.0001,
    base_channels: int = 32,
    dropout: float = 0.2,
    decision_threshold: float = 0.5,
    seed: int = 42,
    num_workers: int = 2,
    device_name: str = "auto",
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Main training workflow for FloodSense AI baseline.
    """
    set_seed(seed)

    # Hardware selection
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print("\n" + "=" * 65)
    print(" FLOODSENSE AI - BASELINE U-NET TRAINING PIPELINE")
    print("=" * 65)
    print(f" Compute Device       : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Epochs               : {epochs}")
    print(f" Batch Size           : {batch_size}")
    print(f" Learning Rate        : {lr}")
    print(f" Base Channels        : {base_channels}")
    print(f" Decision Threshold   : {decision_threshold}")
    print(f" Random Seed          : {seed}")
    print(f" Mode                 : {'SMOKE TEST' if smoke_test else 'FULL TRAINING'}")

    # Build DataLoaders
    train_loader, val_loader, norm_stats = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        pin_memory=(device.type == "cuda"),
    )

    print(f" Train Batches        : {len(train_loader)} ({len(train_loader.dataset)} chips)")
    print(f" Validation Batches   : {len(val_loader)} ({len(val_loader.dataset)} chips)")
    print(f" Normalization (Train): VV mean={norm_stats['vv_mean']:.2f}, std={norm_stats['vv_std']:.2f} | VH mean={norm_stats['vh_mean']:.2f}, std={norm_stats['vh_std']:.2f}")

    # Build Model
    model = get_model(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Model Parameters     : {total_params:,} trainable weights")

    # Loss and Optimizer
    criterion = CombinedMaskedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Directories
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"unet_baseline_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou = -1.0
    best_epoch = -1
    history = []

    print("\nStarting training loop...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<9} | {'Val IoU':<8} | {'Val F1':<7} | {'Val Prec':<8} | {'Val Rec':<7} | {'Time (s)':<8}")
    print("-" * 75)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion, device, threshold=decision_threshold)
        scheduler.step()
        elapsed = time.time() - t0

        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1"]
        val_prec = val_metrics["precision"]
        val_rec = val_metrics["recall"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_seconds": elapsed,
            "confusion_matrix": {
                "tp": val_metrics["tp"],
                "fp": val_metrics["fp"],
                "fn": val_metrics["fn"],
                "tn": val_metrics["tn"],
            },
        }
        history.append(record)

        print(
            f"{epoch:<6} | {train_loss:<10.4f} | {val_loss:<9.4f} | "
            f"{val_iou:<8.4f} | {val_f1:<7.4f} | {val_prec:<8.4f} | "
            f"{val_rec:<7.4f} | {elapsed:<8.1f}"
        )

        # Save best model checkpoint based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            best_checkpoint_path = checkpoints_dir / "unet_baseline_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                    "val_metrics": val_metrics,
                    "norm_stats": norm_stats,
                    "hyperparameters": {
                        "in_channels": 2,
                        "out_channels": 1,
                        "base_channels": base_channels,
                        "dropout": dropout,
                        "threshold": decision_threshold,
                    },
                },
                best_checkpoint_path,
            )

        # If smoke test, break after 1 epoch
        if smoke_test and epoch >= 1:
            print("\n[SMOKE TEST] Exiting early after 1 epoch as requested.")
            break

    total_training_time = time.time() - start_total_time

    # Save final run history and metadata
    history_file = run_dir / "training_history.json"
    summary_report = {
        "best_epoch": best_epoch,
        "best_val_iou": best_val_iou,
        "total_training_time_seconds": total_training_time,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "base_channels": base_channels,
            "dropout": dropout,
            "decision_threshold": decision_threshold,
            "seed": seed,
            "device": str(device),
        },
        "normalization_statistics": norm_stats,
        "history": history,
    }

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    # Save latest checkpoint as well
    latest_checkpoint_path = checkpoints_dir / "unet_baseline_latest.pt"
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "val_iou": history[-1]["val_iou"] if history else 0.0,
            "norm_stats": norm_stats,
        },
        latest_checkpoint_path,
    )

    print("\n" + "=" * 65)
    print(" TRAINING COMPLETE")
    print("=" * 65)
    print(f" Best Validation IoU  : {best_val_iou:.4f} (at epoch {best_epoch})")
    print(f" Best Checkpoint Saved: {checkpoints_dir / 'unet_baseline_best.pt'}")
    print(f" Training Run Artifacts: {run_dir}")
    print(f" Total Time Elapsed   : {total_training_time:.1f} seconds\n")

    return summary_report


def main():
    parser = argparse.ArgumentParser(description="Train FloodSense AI U-Net Baseline")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--checkpoints-dir", type=str, default="models/checkpoints", help="Directory for checkpoints")
    parser.add_argument("--runs-dir", type=str, default="experiments/runs", help="Directory for experiment runs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001, help="AdamW weight decay")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channel count")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability in bottleneck")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probability")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch smoke test")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = (base_dir / args.data_dir).resolve()
    checkpoints_dir = (base_dir / args.checkpoints_dir).resolve()
    runs_dir = (base_dir / args.runs_dir).resolve()

    run_training(
        data_dir=data_dir,
        checkpoints_dir=checkpoints_dir,
        runs_dir=runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        base_channels=args.base_channels,
        dropout=args.dropout,
        decision_threshold=args.threshold,
        seed=args.seed,
        num_workers=args.workers,
        device_name=args.device,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
