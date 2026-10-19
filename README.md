<<<<<<< HEAD
# FloodSense AI: Satellite-Based Flood Mapping and Impact Assessment

FloodSense AI is an end-to-end, research-grade geospatial machine learning project designed to detect and segment **observed flood inundation** from Sentinel-1 Synthetic Aperture Radar (SAR) imagery and aligned geographic covariates.

> **Scope Note:** This project focuses on **observed surface inundation mapping** from satellite acquisitions, not future hydrologic flood forecasting.

---

## 1. System Architecture & Components

```
FloodSense AI/
├── data/                      # Data storage (git-ignored)
│   ├── raw/                   # Raw satellite scenes (e.g., GeoTIFFs)
│   ├── processed/             # Cleaned, standardized 256x256 image patches
│   ├── masks/                 # Ground-truth pixel-level binary flood masks
│   └── samples/               # Demo tiles for web inference verification
├── configs/                   # Experiment and pipeline configurations
│   └── config.yaml            # Master configuration (paths, hyperparameters, splits)
├── models/                    # Model architecture definitions & weights
│   └── checkpoints/           # Trained PyTorch weights (.pt, ignored by git)
├── experiments/               # Experiment tracking and logs
│   └── runs/                  # TensorBoard / CSV metrics and confusion matrices
├── backend/                   # FastAPI REST API
│   ├── app/                   # Application code (main.py, schemas, inference)
│   └── tests/                 # Unit tests (pytest)
├── frontend/                  # Next.js (TypeScript + Tailwind CSS + Leaflet)
├── scripts/                   # Utility and verification scripts
│   └── verify_environment.py  # Hardware & dependency health check
├── reports/                   # Technical reports, figures, and model cards
│   └── figures/               # Generated evaluation plots and flood masks
├── requirements.txt           # Python dependencies pinned for Python 3.11 & CUDA 12.4
├── .gitignore                 # Excludes raw data, weights, caches, and node_modules
└── README.md                  # Project documentation & runbook
```

---

## 2. Environment & Hardware Specifications

- **Operating System:** Windows 11 (64-bit AMD64)
- **Python Environment:** Python 3.11 (`.venv`)
- **PyTorch:** `2.6.0+cu124` (CUDA 12.4 acceleration enabled)
- **GPU Acceleration:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM, Compute Capability 8.9)
- **Node.js Environment:** v22.20.0, npm 10.9.3
- **Geospatial Stack:** `rasterio`, `shapely`, `geopandas`, `pyproj`
- **Backend Stack:** `fastapi`, `uvicorn`, `pydantic`, `pytest`

---

## 3. Quick Start & Setup

### 3.1 Python Virtual Environment

```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install core dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.2 Verify Hardware & Environment

Run the verification script to confirm CUDA GPU detection and package integrity:

```powershell
python scripts/verify_environment.py
```

### 3.3 Run Backend Tests & API Server

```powershell
# Run unit tests
pytest backend/tests

# Start the FastAPI backend server (port 8000)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive OpenAPI docs will be available at: `http://127.0.0.1:8000/docs`

### 3.4 Start Frontend Development Server

```powershell
cd frontend
npm run dev
```

The web dashboard will be available at: `http://localhost:3000`

---

## 4. Planned Research Methodology & Baselines

1. **Phase 2 — Dataset Investigation:** Deep dive into Sentinel-1 SAR flood benchmarks (SenFloods11), analyzing polarizations (VV, VH), no-data masks (-1), and label semantics.
2. **Phase 3 — Progressive Baselines:**
   - **Baseline A:** Classical SAR amplitude thresholding (Otsu / empirical dB cutoffs on VV).
   - **Baseline B:** Pixel-level Random Forest classifier (VV, VH, polarimetric ratios).
   - **Baseline C:** PyTorch U-Net with combined Binary Cross-Entropy + Dice Loss.
3. **Phase 4 — Controlled Research Experiment:** Investigate whether auxiliary geospatial layers (e.g., Copernicus DEM elevation) statistically improve flood segmentation over SAR-only inputs under a strict scene/event-level spatial split.
4. **Phase 5 & 6 — Full-Stack Interactive App & API:** Georeferenced raster upload, interactive Leaflet mapping, inundated area estimation ($\text{km}^2$), and automated assessment reports.

---

## 5. Development Principles & Reproducibility

- **No Data Leakage:** Splits are partitioned strictly by flood event or scene ID—never by randomly shuffling overlapping spatial patches.
- **Metric Integrity:** Precision, Recall, F1, and IoU (Jaccard Index) are calculated strictly excluding invalid/no-data pixels.
- **Responsible Impact Assessment:** Inundated surface area is calculated based on exact pixel spatial resolution and CRS projection properties.
=======
# FloodSenseAI
Satellite-based flood inundation mapping using Sentinel-1 SAR imagery and U-Net deep learning, with geospatial visualization, flood-area estimation, and an interactive web dashboard.
>>>>>>> 251bd52494ee8d1d85c224a7dcad1506b090fcb4
