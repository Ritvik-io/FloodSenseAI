"""
FloodSense AI - Prediction & Error Analysis Visualizer
Generates multi-panel figures for validation scenes showing:
  1. Sentinel-1 VV Backscatter (dB)
  2. Sentinel-1 VH Backscatter (dB)
  3. Ground-Truth Dynamic Flood Mask (Land vs Flood vs No-Data)
  4. Model Predicted Flood Probability Heatmap
  5. Predicted Binary Mask (Threshold = 0.5)
  6. Spatial Error Map (TP=Green, TN=Sand, FP=Amber, FN=Red, NoData=Slate)
"""

import sys
import os
import csv
import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Ensure workspace root in path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models.predict import FloodPredictor


def visualize_prediction(
    predictor: FloodPredictor,
    s1_path: Path,
    lbl_path: Path,
    jrc_path: Optional[Path],
    output_path: Path,
):
    """
    Renders a 6-panel prediction and error analysis figure for a single tile.
    """
    with rasterio.open(s1_path) as s1_src:
        s1_data = s1_src.read()
        vv = s1_data[0].astype(np.float32)
        vh = s1_data[1].astype(np.float32)
        bounds = s1_src.bounds
        crs = s1_src.crs

    with rasterio.open(lbl_path) as lbl_src:
        lbl_data = lbl_src.read(1)

    has_jrc = (jrc_path is not None) and jrc_path.exists()
    jrc_data = np.zeros_like(lbl_data, dtype=np.uint8)
    if has_jrc:
        with rasterio.open(jrc_path) as jrc_src:
            jrc_data = jrc_src.read(1)

    chip_name = s1_path.stem.replace("_S1Hand", "")

    # Run inference
    pred_res = predictor.predict_array(s1_data)
    probs = pred_res["probabilities"]
    pred_mask = pred_res["binary_mask"]
    sar_valid = pred_res["valid_mask"]

    # Overall validity mask: SAR valid AND ground truth valid (LabelHand != -1)
    eval_valid = sar_valid & (lbl_data != -1)

    # Dynamic flood ground-truth target
    gt_flood = ((lbl_data == 1) & (jrc_data == 0)).astype(np.uint8)

    # Binary prediction on valid evaluation pixels
    pred_flood = (pred_mask == 1).astype(np.uint8)

    # Compute tile-level confusion matrix
    tp = int(np.sum((pred_flood == 1) & (gt_flood == 1) & eval_valid))
    fp = int(np.sum((pred_flood == 1) & (gt_flood == 0) & eval_valid))
    fn = int(np.sum((pred_flood == 0) & (gt_flood == 1) & eval_valid))
    tn = int(np.sum((pred_flood == 0) & (gt_flood == 0) & eval_valid))
    nodata_cnt = int(np.sum(~eval_valid))

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Build Error Map
    # Codes: 0 = True Negative, 1 = True Positive, 2 = False Positive, 3 = False Negative, 4 = NoData
    error_map = np.full(lbl_data.shape, 4, dtype=np.int32)
    error_map[(pred_flood == 0) & (gt_flood == 0) & eval_valid] = 0  # TN
    error_map[(pred_flood == 1) & (gt_flood == 1) & eval_valid] = 1  # TP
    error_map[(pred_flood == 1) & (gt_flood == 0) & eval_valid] = 2  # FP
    error_map[(pred_flood == 0) & (gt_flood == 1) & eval_valid] = 3  # FN

    # Plotting setup
    fig = plt.figure(figsize=(22, 12), facecolor="#0B132B")
    plt.rcParams["text.color"] = "#F8FAFC"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#64748B"
    plt.rcParams["ytick.color"] = "#64748B"

    fig.suptitle(
        f"FloodSense AI - Model Prediction & Error Audit | Tile: {chip_name}",
        fontsize=18,
        fontweight="bold",
        color="#38BDF8",
        y=0.98,
    )

    # Panel 1: S1 VV Backscatter (dB)
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(vv, cmap="gray", vmin=-25, vmax=0)
    ax1.set_title("1. Sentinel-1 VV Polarized (dB)", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#94A3B8")
    cbar1.set_label("dB", color="#94A3B8", fontsize=10)

    # Panel 2: S1 VH Backscatter (dB)
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(vh, cmap="gray", vmin=-32, vmax=-5)
    ax2.set_title("2. Sentinel-1 VH Polarized (dB)", fontsize=13, fontweight="bold", pad=8)
    ax2.axis("off")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#94A3B8")
    cbar2.set_label("dB", color="#94A3B8", fontsize=10)

    # Panel 3: Ground-Truth Dynamic Flood Mask
    ax3 = fig.add_subplot(2, 3, 3)
    # Mapping: 0=NoData, 1=Land, 2=Dynamic Flood
    gt_display = np.zeros_like(lbl_data, dtype=int)
    gt_display[~eval_valid] = 0
    gt_display[eval_valid & (gt_flood == 0)] = 1
    gt_display[eval_valid & (gt_flood == 1)] = 2

    cmap_gt = ListedColormap(["#334155", "#E2D9C8", "#0EA5E9"])
    norm_gt = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_gt.N)
    ax3.imshow(gt_display, cmap=cmap_gt, norm=norm_gt)
    ax3.set_title("3. Ground-Truth Dynamic Flood", fontsize=13, fontweight="bold", pad=8)
    ax3.axis("off")
    patches_gt = [
        mpatches.Patch(color="#334155", label=f"No-Data / Cloud ({nodata_cnt:,})"),
        mpatches.Patch(color="#E2D9C8", label=f"Land ({tn + fp:,})"),
        mpatches.Patch(color="#0EA5E9", label=f"Flood Extent ({tp + fn:,})"),
    ]
    ax3.legend(handles=patches_gt, loc="lower right", facecolor="#1E293B", edgecolor="#475569", fontsize=9)

    # Panel 4: Model Predicted Flood Probability Heatmap
    ax4 = fig.add_subplot(2, 3, 4)
    # Mask invalid pixels in heatmap
    probs_display = np.ma.masked_where(~eval_valid, probs)
    im4 = ax4.imshow(probs_display, cmap="Blues_r", vmin=0.0, vmax=1.0)
    # Under-overlay for invalid pixels
    ax4.imshow(np.where(~eval_valid, 1, 0), cmap=ListedColormap(["none", "#1E293B"]), alpha=0.9)
    ax4.set_title("4. Model Predicted Flood Probability", fontsize=13, fontweight="bold", pad=8)
    ax4.axis("off")
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    cbar4.ax.tick_params(colors="#94A3B8")
    cbar4.set_label("Inundation Probability [0.0 - 1.0]", color="#94A3B8", fontsize=10)

    # Panel 5: Predicted Binary Mask (tau = 0.5)
    ax5 = fig.add_subplot(2, 3, 5)
    pred_display = np.zeros_like(lbl_data, dtype=int)
    pred_display[~eval_valid] = 0
    pred_display[eval_valid & (pred_flood == 0)] = 1
    pred_display[eval_valid & (pred_flood == 1)] = 2

    cmap_pred = ListedColormap(["#334155", "#E2D9C8", "#22D3EE"])
    norm_pred = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_pred.N)
    ax5.imshow(pred_display, cmap=cmap_pred, norm=norm_pred)
    ax5.set_title(f"5. Predicted Binary Mask (Threshold = {predictor.threshold})", fontsize=13, fontweight="bold", pad=8)
    ax5.axis("off")
    patches_pred = [
        mpatches.Patch(color="#334155", label="No-Data"),
        mpatches.Patch(color="#E2D9C8", label=f"Predicted Land ({tn + fn:,})"),
        mpatches.Patch(color="#22D3EE", label=f"Predicted Flood ({tp + fp:,})"),
    ]
    ax5.legend(handles=patches_pred, loc="lower right", facecolor="#1E293B", edgecolor="#475569", fontsize=9)

    # Panel 6: Spatial Error Map & Confusion Analysis
    ax6 = fig.add_subplot(2, 3, 6)
    cmap_err = ListedColormap(["#E2D9C8", "#10B981", "#F59E0B", "#EF4444", "#334155"])
    norm_err = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap_err.N)
    ax6.imshow(error_map, cmap=cmap_err, norm=norm_err)
    ax6.set_title("6. Error Map & Confusion Analysis", fontsize=13, fontweight="bold", pad=8)
    ax6.axis("off")

    patches_err = [
        mpatches.Patch(color="#10B981", label=f"True Positive (TP): {tp:,}"),
        mpatches.Patch(color="#E2D9C8", label=f"True Negative (TN): {tn:,}"),
        mpatches.Patch(color="#F59E0B", label=f"False Positive (FP): {fp:,}"),
        mpatches.Patch(color="#EF4444", label=f"False Negative (FN): {fn:,}"),
        mpatches.Patch(color="#334155", label="No-Data / Invalid"),
    ]
    ax6.legend(handles=patches_err, loc="lower right", facecolor="#1E293B", edgecolor="#475569", fontsize=9)

    # Inset metrics box in panel 6
    summary_metrics_text = (
        f"TILE EVALUATION METRICS\n"
        f"IoU (Jaccard) : {iou:.4f}\n"
        f"F1 (Dice)     : {f1:.4f}\n"
        f"Precision     : {precision:.4f}\n"
        f"Recall        : {recall:.4f}"
    )
    ax6.text(
        0.03,
        0.97,
        summary_metrics_text,
        transform=ax6.transAxes,
        fontsize=9.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.2),
        color="#F8FAFC",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  [OK] Saved prediction visualization: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FloodSense AI Predictions & Errors")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/unet_baseline_best.pt", help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--split", type=str, default="flood_valid_data.csv", help="Split CSV to select samples from")
    parser.add_argument("--output-dir", type=str, default="reports/figures/predictions", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to visualize")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    checkpoint_p = (base_dir / args.checkpoint).resolve()
    data_dir_p = (base_dir / args.data_dir).resolve()
    output_dir_p = (base_dir / args.output_dir).resolve()

    print(f"\nInitializing FloodPredictor from {checkpoint_p.name}...")
    predictor = FloodPredictor(checkpoint_path=checkpoint_p)

    csv_path = data_dir_p / "splits" / args.split
    if not csv_path.exists():
        print(f"[ERROR] Split file not found: {csv_path}")
        sys.exit(1)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    # Pick diverse representative chips from different events
    selected_pairs = []
    seen_events = set()
    for row in reader:
        if len(row) >= 2:
            s1_name, lbl_name = row[0].strip(), row[1].strip()
            ev = s1_name.split("_")[0]
            if ev not in seen_events:
                seen_events.add(ev)
                selected_pairs.append((s1_name, lbl_name))
            if len(selected_pairs) >= args.num_samples:
                break

    print(f"Generating error and prediction figures for {len(selected_pairs)} validation scenes...")
    for s1_name, lbl_name in selected_pairs:
        chip_id = s1_name.replace("_S1Hand.tif", "")
        s1_p = data_dir_p / "raw" / "S1Hand" / s1_name
        lbl_p = data_dir_p / "masks" / "LabelHand" / lbl_name
        jrc_p = data_dir_p / "raw" / "JRCWaterHand" / s1_name.replace("_S1Hand.tif", "_JRCWaterHand.tif")
        out_p = output_dir_p / f"prediction_audit_{chip_id}.png"

        visualize_prediction(predictor, s1_p, lbl_p, jrc_p, out_p)

    print(f"\nCompleted! Generated figures saved to: {output_dir_p.resolve()}\n")


if __name__ == "__main__":
    main()
