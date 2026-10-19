"""
FloodSense AI - FastAPI Backend Service
Provides endpoints for health checks, model metadata, satellite image inference, and evaluation reports.
Integrates the trained PyTorch U-Net baseline with geospatial preservation.
"""

import sys
import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure workspace root in path
base_dir = Path(__file__).resolve().parent.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from models.predict import FloodPredictor

app = FastAPI(
    title="FloodSense AI API",
    description="Satellite-Based Flood Mapping and Inundation Assessment Service",
    version="1.0.0",
)

# CORS configuration
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lazy-loaded predictor instance
_predictor_instance: Optional[FloodPredictor] = None


def get_predictor() -> FloodPredictor:
    """Returns the singleton FloodPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        ckpt_path = base_dir / "models" / "checkpoints" / "unet_baseline_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Trained model checkpoint not found at: {ckpt_path}")
        _predictor_instance = FloodPredictor(checkpoint_path=ckpt_path)
    return _predictor_instance


@app.get("/health")
def health_check():
    """
    Health check endpoint returning system status, Python runtime, and GPU capability.
    """
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "status": "healthy",
        "service": "FloodSense AI API",
        "version": "1.0.0",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "compute_device": device_name,
    }


@app.get("/model/info")
def model_info():
    """
    Returns information about supported models, expected input channels, and resolutions.
    """
    return {
        "models": {
            "baseline_threshold": {
                "name": "SAR Amplitude Thresholding",
                "type": "Classical Heuristic",
                "inputs": ["VV (dB)"],
                "status": "configured",
            },
            "baseline_rf": {
                "name": "Random Forest Pixel Classifier",
                "type": "Supervised Classical ML",
                "inputs": ["VV (dB)", "VH (dB)", "Polarimetric Ratio (VH/VV)"],
                "status": "configured",
            },
            "unet_sar": {
                "name": "PyTorch U-Net (SAR-only)",
                "type": "Deep Convolutional Neural Network",
                "inputs": ["VV (dB)", "VH (dB)"],
                "patch_size": [512, 512],
                "status": "active_best_checkpoint",
            },
        },
        "default_model": "unet_sar",
        "output_format": "Binary segmentation mask (1=Flood, 0=Non-flood, 255=No-data)",
    }


@app.get("/model/active")
def active_model_metadata():
    """
    Returns metadata of the loaded active model checkpoint.
    """
    try:
        predictor = get_predictor()
        return {
            "model_architecture": "UNet",
            "checkpoint_epoch": predictor.checkpoint_epoch,
            "validation_iou": predictor.val_iou,
            "decision_threshold": predictor.threshold,
            "base_channels": predictor.base_channels,
            "device": str(predictor.device),
            "normalization_statistics": predictor.norm_stats,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/samples")
def list_benchmark_samples():
    """
    Returns a curated list of benchmark sample chips across diverse global flood events.
    """
    s1_dir = base_dir / "data" / "raw" / "S1Hand"
    if not s1_dir.exists():
        s1_dir = base_dir / "data" / "samples" / "raw" / "S1Hand"

    if not s1_dir.exists():
        return {"samples": []}

    files = sorted(list(s1_dir.glob("*_S1Hand.tif")))
    samples_list = []
    seen_countries = set()

    # Priority representative chips
    priority_ids = [
        "USA_1010394", "Mekong_1149855", "India_1050276", "Ghana_103272",
        "Spain_1167260", "Nigeria_31096", "Bolivia_103757", "Paraguay_1036814"
    ]

    for pid in priority_ids:
        matching = s1_dir / f"{pid}_S1Hand.tif"
        if matching.exists():
            country = pid.split("_")[0]
            samples_list.append({
                "chip_id": pid,
                "country": country,
                "filename": matching.name,
                "is_featured": True,
            })
            seen_countries.add(pid)

    for f in files:
        cid = f.stem.replace("_S1Hand", "")
        if cid not in seen_countries:
            country = cid.split("_")[0]
            samples_list.append({
                "chip_id": cid,
                "country": country,
                "filename": f.name,
                "is_featured": False,
            })
            seen_countries.add(cid)
        if len(samples_list) >= 25:
            break

    return {"samples": samples_list}


@app.post("/predict/sample/{chip_id}")
def predict_sample_scene(
    chip_id: str,
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """
    Runs flood segmentation on an existing dataset sample chip.
    """
    predictor = get_predictor()
    if threshold is not None:
        predictor.threshold = threshold

    # Locate S1 scene
    s1_candidates = [
        base_dir / "data" / "raw" / "S1Hand" / f"{chip_id}_S1Hand.tif",
        base_dir / "data" / "samples" / "raw" / "S1Hand" / f"{chip_id}_S1Hand.tif",
    ]
    s1_path = next((p for p in s1_candidates if p.exists()), None)
    if not s1_path:
        raise HTTPException(status_code=404, detail=f"Sample chip '{chip_id}' not found.")

    out_dir = base_dir / "data" / "predictions"
    results = predictor.predict_geotiff(input_path=s1_path, output_dir=out_dir, save_rasters=True)

    return {
        "chip_id": chip_id,
        "threshold": results["threshold"],
        "inundated_area_km2": round(results["inundated_area_km2"], 4),
        "flood_percentage": round(results["flood_percentage"], 2),
        "flood_pixel_count": results["flood_pixel_count"],
        "valid_pixel_count": results["valid_pixel_count"],
        "total_pixel_count": results["total_pixel_count"],
        "bounds": results["bounds"],
        "lat_lng_bounds": results["lat_lng_bounds"],
        "center": results["center"],
        "crs": results["crs"],
        "overlays": results.get("overlays", {}),
    }


@app.post("/predict/upload")
async def predict_uploaded_scene(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """
    Uploads a Sentinel-1 GeoTIFF and performs flood extent segmentation.
    """
    if not file.filename.endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif, .tiff) files are supported.")

    upload_dir = base_dir / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / file.filename

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predictor = get_predictor()
    if threshold is not None:
        predictor.threshold = threshold

    out_dir = base_dir / "data" / "predictions"
    try:
        results = predictor.predict_geotiff(input_path=temp_path, output_dir=out_dir, save_rasters=True)
        return {
            "filename": file.filename,
            "threshold": results["threshold"],
            "inundated_area_km2": round(results["inundated_area_km2"], 4),
            "flood_percentage": round(results["flood_percentage"], 2),
            "flood_pixel_count": results["flood_pixel_count"],
            "valid_pixel_count": results["valid_pixel_count"],
            "total_pixel_count": results["total_pixel_count"],
            "bounds": results["bounds"],
            "lat_lng_bounds": results["lat_lng_bounds"],
            "center": results["center"],
            "crs": results["crs"],
            "overlays": results.get("overlays", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.get("/metrics/reports")
def get_evaluation_reports():
    """
    Returns stored evaluation reports for the validation and held-out Bolivia splits.
    """
    reports = {}
    val_file = base_dir / "reports" / "val_evaluation_best.json"
    bolivia_file = base_dir / "reports" / "bolivia_evaluation_report.json"

    if val_file.exists():
        with open(val_file, "r", encoding="utf-8") as f:
            reports["validation_split"] = json.load(f)

    if bolivia_file.exists():
        with open(bolivia_file, "r", encoding="utf-8") as f:
            reports["bolivia_heldout_split"] = json.load(f)

    return reports
