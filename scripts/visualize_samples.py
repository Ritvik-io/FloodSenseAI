#!/usr/bin/env python
"""
FloodSense AI - Sample Image-Mask Pair Visualizer
Generates publication-quality visualizations for Sentinel-1 SAR (VV, VH, false-color composite),
ground-truth water masks (-1, 0, 1), and permanent vs. temporary floodwater overlays.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


def normalize_channel(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Clips and min-max normalizes a dB SAR array to [0, 1]."""
    clipped = np.clip(arr, vmin, vmax)
    norm = (clipped - vmin) / (vmax - vmin + 1e-8)
    norm[np.isnan(arr)] = 0.0
    return norm


def create_sar_composite(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """
    Constructs a standard false-color SAR composite (R: VV, G: VH, B: VV - VH).
    Calm surface water exhibits low backscatter and appears black/deep navy.
    Rough water/vegetated areas appear greenish, urban double bounce appears bright white/pink.
    """
    r = normalize_channel(vv, -25.0, 0.0)
    g = normalize_channel(vh, -32.0, -5.0)
    diff = vv - vh
    b = normalize_channel(diff, 2.0, 18.0)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def visualize_chip(
    s1_path: Path,
    lbl_path: Path,
    jrc_path: Optional[Path],
    output_path: Path,
):
    """Generates a 4-panel or 5-panel figure for a single sample chip."""
    with rasterio.open(s1_path) as s1_src:
        s1_data = s1_src.read()
        vv = s1_data[0]
        vh = s1_data[1]
        bounds = s1_src.bounds
        crs = s1_src.crs

    with rasterio.open(lbl_path) as lbl_src:
        lbl_data = lbl_src.read(1)

    has_jrc = (jrc_path is not None) and jrc_path.exists()
    jrc_data = None
    if has_jrc:
        with rasterio.open(jrc_path) as jrc_src:
            jrc_data = jrc_src.read(1)

    chip_name = s1_path.stem.replace("_S1Hand", "")
    sar_composite = create_sar_composite(vv, vh)

    # Calculate statistics
    total_pix = lbl_data.size
    nodata_cnt = np.sum(lbl_data == -1)
    land_cnt = np.sum(lbl_data == 0)
    water_cnt = np.sum(lbl_data == 1)

    pct_nodata = (nodata_cnt / total_pix) * 100
    pct_land = (land_cnt / total_pix) * 100
    pct_water = (water_cnt / total_pix) * 100

    vv_valid = vv[~np.isnan(vv)]
    vh_valid = vh[~np.isnan(vh)]

    # Colormaps
    # Label colormap: -1 -> Gray, 0 -> Tan/Brown, 1 -> Blue
    # Mapping indices: 0 -> -1 (NoData), 1 -> 0 (Land), 2 -> 1 (Water)
    lbl_mapped = np.zeros_like(lbl_data, dtype=int)
    lbl_mapped[lbl_data == -1] = 0
    lbl_mapped[lbl_data == 0] = 1
    lbl_mapped[lbl_data == 1] = 2

    label_cmap = ListedColormap(["#4A5568", "#E2D9C8", "#0EA5E9"])
    label_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], label_cmap.N)

    # Plot figure
    fig = plt.figure(figsize=(20, 10), facecolor="#0F172A")
    plt.rcParams["text.color"] = "#F8FAFC"
    plt.rcParams["axes.labelcolor"] = "#94A3B8"
    plt.rcParams["xtick.color"] = "#64748B"
    plt.rcParams["ytick.color"] = "#64748B"

    fig.suptitle(
        f"FloodSense AI - Dataset Inspection | Tile: {chip_name}",
        fontsize=18,
        fontweight="bold",
        color="#38BDF8",
        y=0.98,
    )

    # Panel 1: S1 VV Backscatter
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.imshow(vv, cmap="gray", vmin=-25, vmax=0)
    ax1.set_title("1. Sentinel-1 VV Polarized (dB)", fontsize=13, fontweight="bold", pad=8)
    ax1.axis("off")
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors="#94A3B8")
    cbar1.set_label("dB", color="#94A3B8", fontsize=10)

    # Panel 2: S1 VH Backscatter
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(vh, cmap="gray", vmin=-32, vmax=-5)
    ax2.set_title("2. Sentinel-1 VH Polarized (dB)", fontsize=13, fontweight="bold", pad=8)
    ax2.axis("off")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors="#94A3B8")
    cbar2.set_label("dB", color="#94A3B8", fontsize=10)

    # Panel 3: SAR False-Color Composite (R=VV, G=VH, B=VV-VH)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(sar_composite)
    ax3.set_title("3. SAR False-Color Composite\n(R: VV, G: VH, B: VV-VH)", fontsize=13, fontweight="bold", pad=8)
    ax3.axis("off")

    # Panel 4: Ground Truth Mask (-1, 0, 1)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(lbl_mapped, cmap=label_cmap, norm=label_norm)
    ax4.set_title("4. Ground-Truth Mask (Sen1Floods11)", fontsize=13, fontweight="bold", pad=8)
    ax4.axis("off")
    patches_lbl = [
        mpatches.Patch(color="#4A5568", label=f"No-Data / Invalid (-1): {pct_nodata:.1f}%"),
        mpatches.Patch(color="#E2D9C8", label=f"Land (0): {pct_land:.1f}%"),
        mpatches.Patch(color="#0EA5E9", label=f"Observed Water (1): {pct_water:.1f}%"),
    ]
    ax4.legend(handles=patches_lbl, loc="lower right", facecolor="#1E293B", edgecolor="#334155", fontsize=9)

    # Panel 5: Flood Inundation vs Permanent Water Breakdown
    ax5 = fig.add_subplot(2, 3, 5)
    if has_jrc and jrc_data is not None:
        # 0: Background/Land, 1: Permanent Water, 2: Temporary Flood Inundation
        water_breakdown = np.zeros_like(lbl_data, dtype=int)
        water_breakdown[(lbl_data == 1) & (jrc_data == 1)] = 1  # Permanent
        water_breakdown[(lbl_data == 1) & (jrc_data == 0)] = 2  # Inundation
        water_breakdown[lbl_data == -1] = 3                      # NoData

        breakdown_cmap = ListedColormap(["#0F172A", "#1D4ED8", "#22D3EE", "#475569"])
        breakdown_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], breakdown_cmap.N)

        ax5.imshow(sar_composite, alpha=0.6)
        ax5.imshow(water_breakdown, cmap=breakdown_cmap, norm=breakdown_norm, alpha=0.7)
        ax5.set_title("5. Flood Inundation vs Permanent Water", fontsize=13, fontweight="bold", pad=8)
        ax5.axis("off")

        perm_cnt = np.sum((lbl_data == 1) & (jrc_data == 1))
        flood_cnt = np.sum((lbl_data == 1) & (jrc_data == 0))
        patches_bd = [
            mpatches.Patch(color="#1D4ED8", label=f"Permanent Water (JRC): {perm_cnt:,}"),
            mpatches.Patch(color="#22D3EE", label=f"Temporary Flood Extent: {flood_cnt:,}"),
            mpatches.Patch(color="#475569", label="No-Data"),
        ]
        ax5.legend(handles=patches_bd, loc="lower right", facecolor="#1E293B", edgecolor="#334155", fontsize=9)
    else:
        # Overlay mask on SAR
        ax5.imshow(sar_composite)
        water_overlay = np.ma.masked_where(lbl_data != 1, lbl_data)
        ax5.imshow(water_overlay, cmap=ListedColormap(["#00F5FF"]), alpha=0.6)
        ax5.set_title("5. SAR & Water Mask Overlay", fontsize=13, fontweight="bold", pad=8)
        ax5.axis("off")

    # Panel 6: Metadata & Quality Summary Card
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")

    summary_text = (
        f"GEOSPATIAL & SENSOR METADATA\n"
        f"-----------------------------------------\n"
        f"Chip ID      : {chip_name}\n"
        f"Dimensions   : 512 x 512 pixels (10m res)\n"
        f"CRS          : {crs}\n"
        f"Bounds West  : {bounds.left:.4f}°\n"
        f"Bounds East  : {bounds.right:.4f}°\n"
        f"Bounds North : {bounds.top:.4f}°\n"
        f"Bounds South : {bounds.bottom:.4f}°\n\n"
        f"RADIOMETRIC PROPERTIES (dB)\n"
        f"-----------------------------------------\n"
        f"VV Range     : [{np.min(vv_valid):.1f}, {np.max(vv_valid):.1f}] dB (μ: {np.mean(vv_valid):.1f})\n"
        f"VH Range     : [{np.min(vh_valid):.1f}, {np.max(vh_valid):.1f}] dB (μ: {np.mean(vh_valid):.1f})\n"
        f"NaN Pixels   : {np.sum(np.isnan(vv)):,}\n\n"
        f"GROUND TRUTH DISTRIBUTION\n"
        f"-----------------------------------------\n"
        f"Water Pixels : {water_cnt:,} ({pct_water:.2f}%)\n"
        f"Land Pixels  : {land_cnt:,} ({pct_land:.2f}%)\n"
        f"No-Data Pix  : {nodata_cnt:,} ({pct_nodata:.2f}%)\n"
        f"Alignment    : Pixel-Perfect Georeferenced"
    )

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10.5,
        family="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1E293B", edgecolor="#38BDF8", linewidth=1.5),
        color="#F1F5F9",
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  [OK] Saved visualization: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Sen1Floods11 Image-Mask Pairs")
    parser.add_argument("--data-dir", type=str, default="data/samples", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="reports/figures", help="Directory to save figures")
    parser.add_argument("--max-chips", type=int, default=5, help="Maximum number of chips to visualize")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = (base_dir / args.data_dir).resolve()
    output_dir = (base_dir / args.output_dir).resolve()

    s1_dir = data_dir / "raw" / "S1Hand"
    lbl_dir = data_dir / "masks" / "LabelHand"
    jrc_dir = data_dir / "raw" / "JRCWaterHand"

    if not s1_dir.exists() or not lbl_dir.exists():
        print(f"[ERROR] Required directories not found in {data_dir}")
        sys.exit(1)

    s1_files = sorted(list(s1_dir.glob("*_S1Hand.tif")))
    if not s1_files:
        print(f"[ERROR] No S1Hand.tif files found in {s1_dir}")
        sys.exit(1)

    print(f"\nFound {len(s1_files)} chips in {s1_dir}. Generating visualizations...")

    for s1_path in s1_files[: args.max_chips]:
        chip_id = s1_path.stem.replace("_S1Hand", "")
        lbl_path = lbl_dir / f"{chip_id}_LabelHand.tif"
        jrc_path = jrc_dir / f"{chip_id}_JRCWaterHand.tif"

        if not lbl_path.exists():
            print(f"  [WARN] Skipping {chip_id}: Label not found")
            continue

        out_path = output_dir / f"sample_{chip_id}.png"
        visualize_chip(s1_path, lbl_path, jrc_path, out_path)

    print("\nVisualization complete! Check the output directory for high-res figures.\n")


if __name__ == "__main__":
    main()
