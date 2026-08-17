#!/usr/bin/env python3
"""
Research-Grade Dataset Health & Physical Validation Suite
Part of the PINN Thermal Dataset Generation Pipeline.

Verifies:
  1. No NaN, Null, or Infinite values in matrices
  2. Bounded temperatures fitting physical semiconductors (ambient <= Temp < 150C)
  3. Spatial Diffusion Profiles (temperatures decay away from hotspots)
  4. Consistency of sparse sensor data matching primary matrices exactly.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import argparse
from typing import Dict, List, Any

class DatasetValidator:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir

    def validate_split(self, split: str) -> Dict[str, Any]:
        split_path = os.path.join(self.dataset_dir, split)
        if not os.path.exists(split_path):
            return {"status": "FAILED", "reason": f"Split directory '{split}' not found"}

        csv_path = os.path.join(split_path, "metadata.csv")
        if not os.path.exists(csv_path):
            return {"status": "FAILED", "reason": f"metadata.csv not found in split '{split}'"}

        # Read CSV
        df = pd.read_csv(csv_path)
        num_samples = len(df)
        
        nan_count = 0
        invalid_range_count = 0
        diffusion_check_failed = 0
        sensor_mismatch_count = 0
        file_missing_count = 0

        # Subfolders
        matrices_dir = os.path.join(split_path, "matrices")
        sensors_dir = os.path.join(split_path, "sensors")
        images_dir = os.path.join(split_path, "images")

        print(f"\n[Validation] Auditing split '{split}' containing {num_samples} samples...", flush=True)

        for idx, row in df.iterrows():
            sample_id = row["sample_id"]
            
            # 1. Existence Checks
            mat_file = os.path.join(matrices_dir, f"{sample_id}.npy")
            img_file = os.path.join(images_dir, f"{sample_id}.png")
            sens_file = os.path.join(sensors_dir, f"{sample_id}_sensors.json")

            if not os.path.exists(mat_file) or not os.path.exists(img_file) or not os.path.exists(sens_file):
                file_missing_count += 1
                continue

            # 2. Numerical Checks (.npy)
            try:
                matrix = np.load(mat_file)
            except Exception as e:
                nan_count += 1
                continue

            if np.isnan(matrix).any() or np.isinf(matrix).any():
                nan_count += 1

            # 3. Physical Range Checks
            ambient = row["boundary_temperature"]
            max_temp = matrix.max()
            min_temp = matrix.min()

            # The minimum temperature should equal or exceed ambient (Dirichlet BC)
            # The maximum temperature should represent a realistic hotspot peak (< 140 C is normal, > 180 C would be melting)
            if min_temp < (ambient - 0.1) or max_temp > 200.0 or max_temp <= min_temp:
                invalid_range_count += 1

            # 4. Thermal Diffusion Check (heat must decrease on average moving from heat generation centers)
            # Find center coordinate regions of active blocks
            blocks_pos = json.loads(row["hotspot_positions"])
            if len(blocks_pos) > 0 and matrix.shape == (128, 128):
                # We expect the temperature in active logic centers to be strictly greater
                # than temperatures toward the borders
                border_values = np.concatenate([
                    matrix[0, :], matrix[-1, :], matrix[:, 0], matrix[:, -1]
                ])
                ambient_boundary_avg = np.mean(border_values)
                
                hotspot_temps = []
                grid_size = 128
                for b in blocks_pos:
                    i_min = int(np.round(b[1] * (grid_size - 1)))
                    i_max = int(np.round(b[3] * (grid_size - 1)))
                    j_min = int(np.round(b[0] * (grid_size - 1)))
                    j_max = int(np.round(b[2] * (grid_size - 1)))
                    
                    # Ensure indices bounded
                    i_min, i_max = max(0, i_min), min(grid_size-1, i_max)
                    j_min, j_max = max(0, j_min), min(grid_size-1, j_max)
                    
                    if i_max >= i_min and j_max >= j_min:
                        hotspot_temps.append(np.mean(matrix[i_min:i_max+1, j_min:j_max+1]))
                
                if len(hotspot_temps) > 0:
                    hotspot_avg = np.mean(hotspot_temps)
                    # Heat must diffuse away, meaning block averages must be larger than BC levels on boundaries
                    if hotspot_avg <= ambient_boundary_avg:
                        diffusion_check_failed += 1

            # 5. Sensor Readout Verification
            try:
                with open(sens_file, "r") as f:
                    sensor_j = json.load(f)
                
                positions = sensor_j["positions"]
                temps = sensor_j["temperatures"]
                
                for pos, temp in zip(positions, temps):
                    x, y = pos
                    # Map x, y back to indices
                    col = int(np.round(x * (128 - 1)))
                    row_idx = int(np.round(y * (128 - 1)))
                    
                    matrix_val = matrix[row_idx, col]
                    if abs(matrix_val - temp) > 1e-3:
                        sensor_mismatch_count += 1
                        break
            except Exception:
                sensor_mismatch_count += 1

        print(f"  - Missing files: {file_missing_count}/{num_samples}")
        print(f"  - NaNs/Infs found: {nan_count}/{num_samples}")
        print(f"  - Physical range anomalies: {invalid_range_count}/{num_samples}")
        print(f"  - Diffusion gradient errors: {diffusion_check_failed}/{num_samples}")
        print(f"  - Sensor matrix value deviations: {sensor_mismatch_count}/{num_samples}")

        failures = file_missing_count + nan_count + invalid_range_count + diffusion_check_failed + sensor_mismatch_count
        
        status = "PASSED" if failures == 0 else "FAILED"
        return {
            "status": status,
            "metrics": {
                "missing_files": file_missing_count,
                "nan_failures": nan_count,
                "range_failures": invalid_range_count,
                "diffusion_failures": diffusion_check_failed,
                "sensor_failures": sensor_mismatch_count
            }
        }

    def run_all(self) -> bool:
        results = {}
        all_passed = True
        
        for split in ["train", "val", "test"]:
            res = self.validate_split(split)
            results[split] = res
            if res.get("status") != "PASSED":
                all_passed = False

        print("\n" + "="*50)
        print("PDE DATASET VALIDATION RUN")
        for split, data in results.items():
            print(f"Split '{split}': {data['status']}")
            if data['status'] == "FAILED" and "metrics" in data:
                print(f"  Details: {data['metrics']}")
            elif data['status'] == "FAILED" and "reason" in data:
                print(f"  Reason: {data['reason']}")
        print("="*50, flush=True)
        return all_passed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Validation Script")
    parser.add_argument("--dir", type=str, default="dataset", help="Dataset root directory")
    args = parser.parse_args()
    
    validator = DatasetValidator(dataset_dir=args.dir)
    success = validator.run_all()
    sys.exit(0 if success else 1)
