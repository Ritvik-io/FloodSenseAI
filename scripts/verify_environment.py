#!/usr/bin/env python
"""
FloodSense AI - Environment Verification Script
Inspects OS, Python, CUDA, GPU acceleration, geospatial stack, and core dependencies.
Runs a dummy tensor allocation on the selected device to confirm GPU compute is functioning.
"""

import sys
import os
import platform
from pathlib import Path


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title.upper()}")
    print("=" * 60)


def check_system():
    print_header("1. System Information")
    print(f"  OS:            {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Machine:       {platform.machine()} ({platform.processor()})")
    print(f"  Python Exec:   {sys.executable}")
    print(f"  Python Ver:    {platform.python_version()} ({platform.architecture()[0]})")


def check_torch_cuda():
    print_header("2. PyTorch & Hardware Acceleration")
    try:
        import torch
        print(f"  PyTorch:       v{torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA Available:{' [YES]' if cuda_available else ' [NO] (CPU only)'}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            capability = torch.cuda.get_device_capability(current_device)
            total_memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            print(f"  Device Count:  {device_count}")
            print(f"  Current Device:CUDA:{current_device} - {device_name}")
            print(f"  Compute Cap:   {capability[0]}.{capability[1]}")
            print(f"  Total VRAM:    {total_memory_gb:.2f} GB")

            # Test tensor allocation and computation on GPU
            x = torch.randn(2, 2, 64, 64, device="cuda")
            y = torch.relu(x)
            assert y.is_cuda, "CUDA tensor check failed"
            print("  GPU Tensor Test: PASSED (Allocated and evaluated on CUDA device)")
        else:
            # Test CPU allocation
            x = torch.randn(2, 2, 64, 64, device="cpu")
            y = torch.relu(x)
            print("  CPU Tensor Test: PASSED (Evaluated on CPU)")

    except ImportError as e:
        print(f"  PyTorch:       [NOT INSTALLED] ({e})")


def check_libraries():
    print_header("3. Core Geospatial & ML Stack")
    packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-Learn"),
        ("matplotlib", "Matplotlib"),
        ("yaml", "PyYAML"),
        ("rasterio", "Rasterio (Geospatial Rasters)"),
        ("shapely", "Shapely (Geometry)"),
        ("geopandas", "GeoPandas (Vector Data)"),
        ("fastapi", "FastAPI (Backend Web Framework)"),
        ("uvicorn", "Uvicorn (ASGI Server)"),
        ("pydantic", "Pydantic (Data Validation)"),
    ]

    for module_name, label in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK]  {label:<32} v{ver}")
        except ImportError:
            print(f"  [--]  {label:<32} (Not installed yet)")


def check_directories():
    print_header("4. Workspace Directories")
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/masks",
        "models/checkpoints",
        "notebooks",
        "experiments/runs",
        "backend/app",
        "backend/tests",
        "configs",
        "scripts",
        "reports/figures",
    ]

    for d in expected_dirs:
        p = base_dir / d
        status = "[EXISTS]" if p.exists() else "[MISSING]"
        print(f"  {status:<10} {d}")


if __name__ == "__main__":
    print("\nStarting FloodSense AI Environment Verification...")
    check_system()
    check_torch_cuda()
    check_libraries()
    check_directories()
    print("\nVerification complete.\n")
