#!/usr/bin/env python
"""
FloodSense AI - Dataset Audit Script
Validates Sen1Floods11 dataset integrity, Sentinel-1 VV/VH imagery,
ground-truth label encoding (-1, 0, 1), permanent water layer, nodata values,
geospatial alignment (CRS, affine transform, dimensions), and class statistics.
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import rasterio
from rasterio.crs import CRS


class DatasetAuditor:
    def __init__(self, data_dir: Path, report_dir: Path):
        self.data_dir = data_dir
        self.report_dir = report_dir
        self.splits_dir = data_dir / "splits"
        self.s1_dir = data_dir / "raw" / "S1Hand"
        self.lbl_dir = data_dir / "masks" / "LabelHand"
        self.jrc_dir = data_dir / "raw" / "JRCWaterHand"

        self.results = {
            "splits": {},
            "total_pairs_checked": 0,
            "passed_pairs": 0,
            "failed_pairs": 0,
            "checks": {
                "file_existence": True,
                "dimensions_512x512": True,
                "s1_bands_vv_vh": True,
                "s1_dtype_float32": True,
                "label_values_valid": True,
                "crs_match": True,
                "affine_transform_match": True,
            },
            "pixel_statistics": {
                "total_pixels": 0,
                "nodata_pixels_minus_1": 0,
                "land_pixels_0": 0,
                "water_pixels_1": 0,
                "permanent_water_pixels": 0,
                "temporary_flood_pixels": 0,
            },
            "sar_statistics": {
                "vv_min_db": float("inf"),
                "vv_max_db": float("-inf"),
                "vh_min_db": float("inf"),
                "vh_max_db": float("-inf"),
                "vv_sum": 0.0,
                "vh_sum": 0.0,
                "valid_pixel_count": 0,
                "nan_pixel_count": 0,
            },
            "failures": [],
            "warnings": [],
        }

    def load_split_csvs(self) -> Dict[str, List[Dict[str, str]]]:
        split_files = [
            "flood_train_data.csv",
            "flood_valid_data.csv",
            "flood_test_data.csv",
            "flood_bolivia_data.csv",
        ]
        splits = {}
        for fname in split_files:
            csv_path = self.splits_dir / fname
            if not csv_path.exists():
                self.results["warnings"].append(f"Split CSV not found: {fname}")
                continue

            pairs = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        pairs.append({"s1": row[0].strip(), "label": row[1].strip()})
            splits[fname] = pairs
            self.results["splits"][fname] = len(pairs)
        return splits

    def audit_pair(self, s1_name: str, lbl_name: str) -> bool:
        s1_path = self.s1_dir / s1_name
        lbl_path = self.lbl_dir / lbl_name
        jrc_name = s1_name.replace("_S1Hand.tif", "_JRCWaterHand.tif")
        jrc_path = self.jrc_dir / jrc_name

        # 1. Existence check
        if not s1_path.exists():
            self.results["failures"].append(f"Missing S1 raster: {s1_name}")
            self.results["checks"]["file_existence"] = False
            return False
        if not lbl_path.exists():
            self.results["failures"].append(f"Missing Label raster: {lbl_name}")
            self.results["checks"]["file_existence"] = False
            return False

        has_jrc = jrc_path.exists()

        try:
            with rasterio.open(s1_path) as s1_src, rasterio.open(lbl_path) as lbl_src:
                # 2. Dimensions check
                if s1_src.width != 512 or s1_src.height != 512:
                    self.results["failures"].append(f"{s1_name}: Invalid dimensions {s1_src.width}x{s1_src.height}")
                    self.results["checks"]["dimensions_512x512"] = False
                    return False
                if lbl_src.width != 512 or lbl_src.height != 512:
                    self.results["failures"].append(f"{lbl_name}: Invalid dimensions {lbl_src.width}x{lbl_src.height}")
                    self.results["checks"]["dimensions_512x512"] = False
                    return False

                # 3. Channels check (VV, VH)
                if s1_src.count != 2:
                    self.results["failures"].append(f"{s1_name}: Expected 2 bands (VV, VH), found {s1_src.count}")
                    self.results["checks"]["s1_bands_vv_vh"] = False
                    return False
                if lbl_src.count != 1:
                    self.results["failures"].append(f"{lbl_name}: Expected 1 band, found {lbl_src.count}")
                    return False

                # 4. Data types
                if s1_src.dtypes[0] != "float32":
                    self.results["warnings"].append(f"{s1_name}: Dtype is {s1_src.dtypes[0]}, expected float32")

                # 5. Coordinate Reference System (CRS) check
                if s1_src.crs != lbl_src.crs:
                    self.results["failures"].append(f"{s1_name} and {lbl_name}: CRS mismatch ({s1_src.crs} != {lbl_src.crs})")
                    self.results["checks"]["crs_match"] = False
                    return False

                # 6. Affine transform / georeferencing alignment check
                t1 = s1_src.transform
                t2 = lbl_src.transform
                if not (
                    np.isclose(t1.a, t2.a, atol=1e-5)
                    and np.isclose(t1.b, t2.b, atol=1e-5)
                    and np.isclose(t1.c, t2.c, atol=1e-5)
                    and np.isclose(t1.d, t2.d, atol=1e-5)
                    and np.isclose(t1.e, t2.e, atol=1e-5)
                    and np.isclose(t1.f, t2.f, atol=1e-5)
                ):
                    self.results["failures"].append(f"{s1_name} and {lbl_name}: Affine transform misaligned")
                    self.results["checks"]["affine_transform_match"] = False
                    return False

                # 7. Label Values & Encoding Verification
                lbl_arr = lbl_src.read(1)
                unique_vals = set(np.unique(lbl_arr))
                invalid_vals = unique_vals - {-1, 0, 1}
                if invalid_vals:
                    self.results["failures"].append(f"{lbl_name}: Unexpected label values {invalid_vals}")
                    self.results["checks"]["label_values_valid"] = False
                    return False

                # Pixel counts
                nodata_cnt = int(np.sum(lbl_arr == -1))
                land_cnt = int(np.sum(lbl_arr == 0))
                water_cnt = int(np.sum(lbl_arr == 1))

                stats = self.results["pixel_statistics"]
                stats["total_pixels"] += int(lbl_arr.size)
                stats["nodata_pixels_minus_1"] += nodata_cnt
                stats["land_pixels_0"] += land_cnt
                stats["water_pixels_1"] += water_cnt

                # Permanent water analysis
                if has_jrc:
                    with rasterio.open(jrc_path) as jrc_src:
                        jrc_arr = jrc_src.read(1)
                        perm_water_cnt = int(np.sum((lbl_arr == 1) & (jrc_arr == 1)))
                        temp_flood_cnt = int(np.sum((lbl_arr == 1) & (jrc_arr == 0)))
                        stats["permanent_water_pixels"] += perm_water_cnt
                        stats["temporary_flood_pixels"] += temp_flood_cnt

                # 8. SAR Imagery Statistics (VV and VH)
                s1_arr = s1_src.read()
                vv = s1_arr[0]
                vh = s1_arr[1]

                vv_valid = vv[~np.isnan(vv)]
                vh_valid = vh[~np.isnan(vh)]

                sar_stats = self.results["sar_statistics"]
                sar_stats["nan_pixel_count"] += int(np.sum(np.isnan(vv)))

                if len(vv_valid) > 0:
                    sar_stats["vv_min_db"] = min(sar_stats["vv_min_db"], float(np.min(vv_valid)))
                    sar_stats["vv_max_db"] = max(sar_stats["vv_max_db"], float(np.max(vv_valid)))
                    sar_stats["vh_min_db"] = min(sar_stats["vh_min_db"], float(np.min(vh_valid)))
                    sar_stats["vh_max_db"] = max(sar_stats["vh_max_db"], float(np.max(vh_valid)))
                    sar_stats["vv_sum"] += float(np.sum(vv_valid))
                    sar_stats["vh_sum"] += float(np.sum(vh_valid))
                    sar_stats["valid_pixel_count"] += len(vv_valid)

        except Exception as e:
            self.results["failures"].append(f"Error auditing {s1_name}/{lbl_name}: {str(e)}")
            return False

        return True

    def run_audit(self, limit: Optional[int] = None) -> bool:
        print("\n" + "=" * 65)
        print(" FLOODSENSE AI - SEN1FLOODS11 DATASET INTEGRITY AUDIT")
        print("=" * 65)
        print(f" Data Directory : {self.data_dir.resolve()}")

        splits = self.load_split_csvs()
        if not splits:
            print("[FATAL] No split files found. Run the download script first.")
            return False

        all_pairs = []
        for split_name, pairs in splits.items():
            for p in pairs:
                all_pairs.append((p["s1"], p["label"], split_name))

        # Check existing files
        present_pairs = []
        for s1, lbl, sp in all_pairs:
            if (self.s1_dir / s1).exists() and (self.lbl_dir / lbl).exists():
                present_pairs.append((s1, lbl, sp))

        print(f" Total Pairs in Splits : {len(all_pairs)}")
        print(f" Pairs Present on Disk : {len(present_pairs)}")

        if limit is not None and limit > 0:
            target_pairs = present_pairs[:limit]
            print(f" Auditing Limit        : First {limit} present pairs")
        else:
            target_pairs = present_pairs

        if not target_pairs:
            print("[FATAL] No pairs found on disk to audit.")
            return False

        print(f"\nAuditing {len(target_pairs)} image-mask pairs...")

        for idx, (s1, lbl, split_name) in enumerate(target_pairs):
            self.results["total_pairs_checked"] += 1
            passed = self.audit_pair(s1, lbl)
            if passed:
                self.results["passed_pairs"] += 1
            else:
                self.results["failed_pairs"] += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(target_pairs):
                sys.stdout.write(f"\r  Audited: {idx + 1}/{len(target_pairs)} pairs...")
                sys.stdout.flush()

        print("\n\n" + "=" * 65)
        print(" AUDIT RESULTS SUMMARY")
        print("=" * 65)

        for check_name, check_passed in self.results["checks"].items():
            status = "[PASS]" if check_passed else "[FAIL]"
            print(f"  {status:<8} {check_name.replace('_', ' ').title()}")

        p_stats = self.results["pixel_statistics"]
        tot = p_stats["total_pixels"]
        if tot > 0:
            pct_nodata = (p_stats["nodata_pixels_minus_1"] / tot) * 100
            pct_land = (p_stats["land_pixels_0"] / tot) * 100
            pct_water = (p_stats["water_pixels_1"] / tot) * 100

            print("\n  Pixel Distribution:")
            print(f"    - No-Data (-1)  : {p_stats['nodata_pixels_minus_1']:>10,} pixels ({pct_nodata:.2f}%)")
            print(f"    - Land (0)      : {p_stats['land_pixels_0']:>10,} pixels ({pct_land:.2f}%)")
            print(f"    - Water (1)     : {p_stats['water_pixels_1']:>10,} pixels ({pct_water:.2f}%)")
            if p_stats["permanent_water_pixels"] > 0:
                print(f"      * Permanent Water : {p_stats['permanent_water_pixels']:>10,} pixels")
                print(f"      * Temporary Flood : {p_stats['temporary_flood_pixels']:>10,} pixels")

        s_stats = self.results["sar_statistics"]
        if s_stats["valid_pixel_count"] > 0:
            vv_mean = s_stats["vv_sum"] / s_stats["valid_pixel_count"]
            vh_mean = s_stats["vh_sum"] / s_stats["valid_pixel_count"]
            print("\n  SAR Backscatter (dB):")
            print(f"    - VV Band: Min = {s_stats['vv_min_db']:.2f} dB, Max = {s_stats['vv_max_db']:.2f} dB, Mean = {vv_mean:.2f} dB")
            print(f"    - VH Band: Min = {s_stats['vh_min_db']:.2f} dB, Max = {s_stats['vh_max_db']:.2f} dB, Mean = {vh_mean:.2f} dB")
            print(f"    - NaN Pixels: {s_stats['nan_pixel_count']:,}")

        print(f"\n  Checked: {self.results['total_pairs_checked']} | Passed: {self.results['passed_pairs']} | Failed: {self.results['failed_pairs']}")

        if self.results["failures"]:
            print(f"\n  [!] {len(self.results['failures'])} Failure(s) Detected:")
            for f in self.results["failures"][:10]:
                print(f"      - {f}")
            if len(self.results["failures"]) > 10:
                print(f"      ... and {len(self.results['failures']) - 10} more.")

        # Save JSON audit report
        self.report_dir.mkdir(parents=True, exist_ok=True)
        report_file = self.report_dir / "sen1floods11_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n  Detailed audit report written to: {report_file.resolve()}")

        all_clean = (self.results["failed_pairs"] == 0) and all(self.results["checks"].values())
        if all_clean:
            print("\n>>> AUDIT PASSED: All verified image-mask pairs comply with FloodSense AI standards. <<<\n")
        else:
            print("\n>>> AUDIT FAILED: Discrepancies detected. Review failures before proceeding. <<<\n")

        return all_clean


def main():
    parser = argparse.ArgumentParser(description="Audit Sen1Floods11 Dataset Integrity")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing dataset")
    parser.add_argument("--report-dir", type=str, default="reports", help="Directory for JSON reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit audit to first N present pairs")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = (base_dir / args.data_dir).resolve()
    report_dir = (base_dir / args.report_dir).resolve()

    auditor = DatasetAuditor(data_dir=data_dir, report_dir=report_dir)
    passed = auditor.run_audit(limit=args.limit)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
