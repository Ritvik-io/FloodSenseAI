"""
FloodSense AI - Production Model Inference Module
Loads trained U-Net checkpoint, uses frozen training normalization statistics,
preserves geospatial metadata, and outputs calibrated flood probabilities and binary masks.
"""

import sys
import os
import io
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import rasterio
from PIL import Image
import torch
import torch.nn as nn

# Ensure workspace root in sys.path
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models.unet import get_model


class FloodPredictor:
    """
    Inference engine for FloodSense AI U-Net baseline.
    Loads checkpoint, verifies and uses frozen training set normalization statistics,
    and runs per-pixel segmentation preserving geospatial georeferencing.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/checkpoints/unet_baseline_best.pt",
        device: str = "auto",
        threshold: Optional[float] = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        # Hardware selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)

        # Validate and load normalization statistics strictly from checkpoint
        if "norm_stats" not in ckpt or ckpt["norm_stats"] is None:
            raise KeyError(f"Checkpoint {self.checkpoint_path} lacks 'norm_stats'! Refusing to recompute.")

        self.norm_stats = ckpt["norm_stats"]
        self.vv_mean = float(self.norm_stats["vv_mean"])
        self.vv_std = float(self.norm_stats["vv_std"])
        self.vh_mean = float(self.norm_stats["vh_mean"])
        self.vh_std = float(self.norm_stats["vh_std"])

        # Model hyperparameters
        hyperparams = ckpt.get("hyperparameters", {})
        self.base_channels = hyperparams.get("base_channels", 32)
        self.dropout = hyperparams.get("dropout", 0.2)
        default_thresh = hyperparams.get("threshold", 0.5)
        self.threshold = threshold if threshold is not None else default_thresh

        # Initialize model architecture and load weights
        self.model = get_model(
            in_channels=2,
            out_channels=1,
            base_channels=self.base_channels,
            dropout=self.dropout,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.checkpoint_epoch = ckpt.get("epoch", "unknown")
        self.val_iou = ckpt.get("val_iou", None)

    def preprocess(self, s1_data: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocesses raw Sentinel-1 VV/VH imagery using frozen training statistics.
        Args:
            s1_data: np.ndarray of shape (2, H, W) in decibels (dB).
        Returns:
            image_tensor: torch.Tensor of shape (1, 2, H, W) normalized float32.
            valid_mask: np.ndarray of shape (H, W) bool, True for valid pixels.
        """
        if s1_data.ndim != 3 or s1_data.shape[0] < 2:
            raise ValueError(f"Expected input array of shape (2, H, W), got {s1_data.shape}")

        vv = s1_data[0].astype(np.float32)
        vh = s1_data[1].astype(np.float32)

        # Detect valid SAR pixels (finite, non-NaN)
        valid_mask = (~np.isnan(vv)) & (~np.isnan(vh)) & (~np.isinf(vv)) & (~np.isinf(vh))

        # Fill NaNs/Infs with training set mean so normalized value is 0.0
        vv_clean = np.where(valid_mask, vv, self.vv_mean)
        vh_clean = np.where(valid_mask, vh, self.vh_mean)

        vv_norm = (vv_clean - self.vv_mean) / self.vv_std
        vh_norm = (vh_clean - self.vh_mean) / self.vh_std

        tensor_np = np.stack([vv_norm, vh_norm], axis=0)[np.newaxis, ...]  # (1, 2, H, W)
        image_tensor = torch.from_numpy(tensor_np).float().to(self.device)

        return image_tensor, valid_mask

    @torch.no_grad()
    def predict_array(self, s1_data: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a numpy array of SAR data (2, H, W).
        Returns probability map, binary mask, and summary statistics.
        """
        image_tensor, valid_mask = self.preprocess(s1_data)
        logits = self.model(image_tensor)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()  # (H, W)

        # Create binary mask (0 = land/non-flood, 1 = flood, 255 = invalid/nodata)
        binary_mask = (probs >= self.threshold).astype(np.uint8)
        binary_mask[~valid_mask] = 255

        # Mask invalid probabilities with NaN
        probs_clean = np.where(valid_mask, probs, np.nan).astype(np.float32)

        valid_count = int(np.sum(valid_mask))
        flood_count = int(np.sum(binary_mask == 1))
        flood_pct = (flood_count / max(valid_count, 1)) * 100.0

        return {
            "probabilities": probs_clean,
            "binary_mask": binary_mask,
            "valid_mask": valid_mask,
            "threshold": self.threshold,
            "flood_pixel_count": flood_count,
            "valid_pixel_count": valid_count,
            "total_pixel_count": int(valid_mask.size),
            "flood_percentage": flood_pct,
        }

    def generate_overlay_data_urls(
        self,
        s1_data: np.ndarray,
        binary_mask: np.ndarray,
        probabilities: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Dict[str, str]:
        """
        Converts raster predictions into browser-renderable transparent PNG data URLs
        for Leaflet L.imageOverlay rendering.
        """
        h, w = binary_mask.shape

        # 1. SAR grayscale visualization (VV backscatter [-25, 0] dB)
        vv = s1_data[0]
        norm_vv = np.clip((vv - (-25.0)) / 25.0, 0.0, 1.0) * 255.0
        sar_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        sar_rgba[..., 0] = norm_vv.astype(np.uint8)
        sar_rgba[..., 1] = norm_vv.astype(np.uint8)
        sar_rgba[..., 2] = norm_vv.astype(np.uint8)
        sar_rgba[..., 3] = np.where(valid_mask, 255, 0).astype(np.uint8)

        buf_sar = io.BytesIO()
        Image.fromarray(sar_rgba).save(buf_sar, format="PNG", optimize=True)
        sar_data_url = "data:image/png;base64," + base64.b64encode(buf_sar.getvalue()).decode("utf-8")

        # 2. Binary flood mask overlay (flood = vibrant neon cyan #00F5FF, non-flood/invalid = transparent)
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        is_flood = (binary_mask == 1) & valid_mask
        mask_rgba[is_flood] = [0, 245, 255, 220]

        buf_mask = io.BytesIO()
        Image.fromarray(mask_rgba).save(buf_mask, format="PNG", optimize=True)
        mask_data_url = "data:image/png;base64," + base64.b64encode(buf_mask.getvalue()).decode("utf-8")

        # 3. Continuous probability heatmap overlay
        prob_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        p = np.nan_to_num(probabilities, nan=0.0)
        active = (p >= 0.05) & valid_mask
        if np.any(active):
            prob_rgba[active, 0] = np.clip(p[active] * 180, 0, 255).astype(np.uint8)
            prob_rgba[active, 1] = np.clip(p[active] * 240, 0, 255).astype(np.uint8)
            prob_rgba[active, 2] = 255
            prob_rgba[active, 3] = np.clip(p[active] * 210 + 40, 0, 230).astype(np.uint8)

        buf_prob = io.BytesIO()
        Image.fromarray(prob_rgba).save(buf_prob, format="PNG", optimize=True)
        prob_data_url = "data:image/png;base64," + base64.b64encode(buf_prob.getvalue()).decode("utf-8")

        return {
            "sar_overlay": sar_data_url,
            "mask_overlay": mask_data_url,
            "prob_overlay": prob_data_url,
        }

    def predict_geotiff(
        self,
        input_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        save_rasters: bool = True,
    ) -> Dict[str, Any]:
        """
        Infers flood extent from an input Sentinel-1 GeoTIFF and writes
        georeferenced probability and binary mask GeoTIFFs preserving CRS and Transform.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with rasterio.open(input_path) as src:
            s1_data = src.read()  # (2, H, W)
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform
            bounds = src.bounds
            width = src.width
            height = src.height

        pred_results = self.predict_array(s1_data)

        # Calculate approximate physical area in square kilometers
        # Approximate 10m x 10m pixel = 100 m^2 = 1e-4 km^2
        # If EPSG:4326, calculate latitude-adjusted pixel area
        mean_lat = (bounds.top + bounds.bottom) / 2.0
        deg_to_m = 111320.0
        dx_m = abs(transform.a) * deg_to_m * np.cos(np.radians(mean_lat)) if crs and crs.is_geographic else abs(transform.a)
        dy_m = abs(transform.e) * deg_to_m if crs and crs.is_geographic else abs(transform.e)
        pixel_area_km2 = (dx_m * dy_m) / 1e6

        inundated_km2 = pred_results["flood_pixel_count"] * pixel_area_km2
        pred_results["inundated_area_km2"] = inundated_km2
        pred_results["pixel_area_km2"] = pixel_area_km2
        pred_results["crs"] = str(crs)
        pred_results["bounds"] = {
            "left": float(bounds.left),
            "bottom": float(bounds.bottom),
            "right": float(bounds.right),
            "top": float(bounds.top),
        }
        # Leaflet format [[south, west], [north, east]]
        pred_results["lat_lng_bounds"] = [
            [float(bounds.bottom), float(bounds.left)],
            [float(bounds.top), float(bounds.right)],
        ]
        pred_results["center"] = [
            float((bounds.bottom + bounds.top) / 2.0),
            float((bounds.left + bounds.right) / 2.0),
        ]

        # Generate browser-renderable transparent PNG data URLs for Leaflet
        overlays = self.generate_overlay_data_urls(
            s1_data=s1_data,
            binary_mask=pred_results["binary_mask"],
            probabilities=pred_results["probabilities"],
            valid_mask=pred_results["valid_mask"],
        )
        pred_results["overlays"] = overlays

        # Write output GeoTIFFs preserving exact spatial references
        if save_rasters and output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem.replace("_S1Hand", "")

            # 1. Probability GeoTIFF (float32, nodata=nan)
            prob_path = output_dir / f"{stem}_flood_prob.tif"
            prob_profile = profile.copy()
            prob_profile.update(
                count=1,
                dtype="float32",
                nodata=np.nan,
                compress="deflate",
            )
            with rasterio.open(prob_path, "w", **prob_profile) as dst:
                dst.write(pred_results["probabilities"], 1)
            pred_results["prob_geotiff"] = str(prob_path)

            # 2. Binary Mask GeoTIFF (uint8: 0=land, 1=flood, 255=nodata)
            mask_path = output_dir / f"{stem}_flood_mask.tif"
            mask_profile = profile.copy()
            mask_profile.update(
                count=1,
                dtype="uint8",
                nodata=255,
                compress="deflate",
            )
            with rasterio.open(mask_path, "w", **mask_profile) as dst:
                dst.write(pred_results["binary_mask"], 1)
            pred_results["mask_geotiff"] = str(mask_path)

        return pred_results


def main():
    parser = argparse.ArgumentParser(description="FloodSense AI Sentinel-1 Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input S1 GeoTIFF")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/unet_baseline_best.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/predictions", help="Directory to save output rasters")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    args = parser.parse_args()

    predictor = FloodPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )

    results = predictor.predict_geotiff(
        input_path=args.input,
        output_dir=args.output_dir,
        save_rasters=True,
    )

    print("\n" + "=" * 60)
    print(" FLOODSENSE AI - INFERENCE SUMMARY")
    print("=" * 60)
    print(f" Input Scene         : {args.input}")
    print(f" Decision Threshold  : {results['threshold']}")
    print(f" Valid Pixels        : {results['valid_pixel_count']:,} / {results['total_pixel_count']:,}")
    print(f" Predicted Flood Pix : {results['flood_pixel_count']:,} ({results['flood_percentage']:.2f}%)")
    print(f" Inundated Area      : {results['inundated_area_km2']:.3f} km²")
    if "prob_geotiff" in results:
        print(f" Probability GeoTIFF : {results['prob_geotiff']}")
        print(f" Binary Mask GeoTIFF : {results['mask_geotiff']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
