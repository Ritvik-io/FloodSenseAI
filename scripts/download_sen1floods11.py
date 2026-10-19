#!/usr/bin/env python
"""
FloodSense AI - Sen1Floods11 Hand-Labeled Subset Downloader
Downloads the official hand-labeled train/val/test splits, Sentinel-1 SAR GeoTIFFs,
ground-truth Label masks, and JRC permanent water rasters from Google Cloud Storage.

Storage Base URL: https://storage.googleapis.com/sen1floods11/v1.1/
Hand-labeled subset: 446 chips (~700 MB total)
  - Train: 252 chips (10 events)
  - Val:   89 chips (10 events)
  - Test:  90 chips (10 events)
  - Bolivia (Held-out): 15 chips
"""

import os
import sys
import csv
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

SPLIT_FILES = [
    "flood_train_data.csv",
    "flood_valid_data.csv",
    "flood_test_data.csv",
    "flood_bolivia_data.csv",
]


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> Tuple[bool, str]:
    """Downloads a file if not already present or incomplete."""
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True, "cached"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "FloodSense-AI-Downloader/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(tmp_path, "wb") as out_file:
                chunk_size = 64 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                tmp_path.replace(dest_path)
                return True, "downloaded"
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == max_retries - 1:
                return False, str(e)
    return False, "failed all retries"


def fetch_splits(data_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Downloads split CSV files and parses pairs of (S1Hand, LabelHand)."""
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    parsed_splits = {}
    print("\n[1/3] Downloading & parsing split CSV files...")

    for split_csv in SPLIT_FILES:
        url = f"{GCS_BASE}/splits/flood_handlabeled/{split_csv}"
        dest = splits_dir / split_csv
        ok, status = download_file(url, dest)
        if not ok:
            print(f"  [ERROR] Failed to download {split_csv}: {status}")
            continue

        pairs = []
        with open(dest, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    s1_name, lbl_name = row[0].strip(), row[1].strip()
                    if s1_name and lbl_name:
                        pairs.append((s1_name, lbl_name))
        parsed_splits[split_csv] = pairs
        print(f"  [OK] {split_csv:<24}: {len(pairs):>3} chips ({status})")

    return parsed_splits


def download_dataset(
    data_dir: Path,
    include_jrc: bool = True,
    sample_only: int = 0,
    max_workers: int = 8,
    dry_run: bool = False,
):
    """
    Downloads hand-labeled SAR chips, ground truth masks, and permanent water layers.
    """
    splits = fetch_splits(data_dir)

    all_pairs: List[Tuple[str, str]] = []
    seen = set()
    for split_name, pairs in splits.items():
        for s1, lbl in pairs:
            if s1 not in seen:
                seen.add(s1)
                all_pairs.append((s1, lbl))

    if sample_only > 0:
        all_pairs = all_pairs[:sample_only]
        print(f"\n[INFO] Sample mode active: limiting to first {sample_only} pairs.")

    print(f"\n[2/3] Preparing raster downloads ({len(all_pairs)} unique chips)...")

    s1_dir = data_dir / "raw" / "S1Hand"
    lbl_dir = data_dir / "masks" / "LabelHand"
    jrc_dir = data_dir / "raw" / "JRCWaterHand"

    tasks = []
    for s1_file, lbl_file in all_pairs:
        # 1. S1 SAR
        s1_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S1Hand/{s1_file}"
        tasks.append((s1_url, s1_dir / s1_file, "S1"))

        # 2. Label
        lbl_url = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{lbl_file}"
        tasks.append((lbl_url, lbl_dir / lbl_file, "Label"))

        # 3. JRC Permanent Water
        if include_jrc:
            jrc_file = s1_file.replace("_S1Hand.tif", "_JRCWaterHand.tif")
            jrc_url = f"{GCS_BASE}/data/flood_events/HandLabeled/JRCWaterHand/{jrc_file}"
            tasks.append((jrc_url, jrc_dir / jrc_file, "JRC"))

    print(f"  Total individual GeoTIFF rasters to download: {len(tasks)}")

    if dry_run:
        print("\n[DRY RUN] Download summary:")
        print(f"  Target Directory: {data_dir.resolve()}")
        print(f"  Sentinel-1 SAR chips : {len(all_pairs)} files (~695 MB)")
        print(f"  Ground-truth masks   : {len(all_pairs)} files (~2.5 MB)")
        if include_jrc:
            print(f"  JRC permanent water  : {len(all_pairs)} files (~0.5 MB)")
        print(f"  Total Estimated Size : ~700 MB")
        print("  Dry run complete. No files were downloaded.")
        return

    print(f"\n[3/3] Downloading {len(tasks)} rasters with {max_workers} worker threads...")

    completed = 0
    failed = 0
    cached = 0

    def _worker(item):
        url, path, kind = item
        success, msg = download_file(url, path)
        return success, msg, path.name

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            ok, msg, fname = future.result()
            if ok:
                if msg == "cached":
                    cached += 1
                else:
                    completed += 1
            else:
                failed += 1
                print(f"  [ERROR] {fname}: {msg}")

            done_count = completed + cached + failed
            if done_count % 50 == 0 or done_count == len(tasks):
                sys.stdout.write(
                    f"\r  Progress: {done_count}/{len(tasks)} "
                    f"({(done_count/len(tasks))*100:.1f}%) | "
                    f"New: {completed}, Cached: {cached}, Errors: {failed}"
                )
                sys.stdout.flush()

    print(f"\n\nDownload complete! Total: {len(tasks)} | Downloaded: {completed} | Cached: {cached} | Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="Sen1Floods11 Hand-Labeled Subset Downloader")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--sample-only", type=int, default=0, help="Download only N sample chips for inspection")
    parser.add_argument("--no-jrc", action="store_true", help="Skip downloading JRC permanent water layers")
    parser.add_argument("--dry-run", action="store_true", help="Print summary without downloading rasters")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    target_data_dir = (base_dir / args.data_dir).resolve()

    download_dataset(
        data_dir=target_data_dir,
        include_jrc=not args.no_jrc,
        sample_only=args.sample_only,
        max_workers=args.workers,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
