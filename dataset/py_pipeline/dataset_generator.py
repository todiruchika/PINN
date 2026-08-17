#!/usr/bin/env python3
"""
Orchestrated Dataset Generator for PINN Thermal Projects
Part of the PINN Thermal Dataset Generation Pipeline.

Handles multiprocessing, output formatting, split distribution (train, val, test),
and writes all requested deliverables (.npy, .png, .json, and metadata.csv).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/container environments
import matplotlib.pyplot as plt
import multiprocessing as mp
from typing import List, Dict, Any, Tuple

# Import local engines
from floorplan_generator import FloorplanGenerator
from pde_solver import ThermalPDESolver
from sensor_generator import SensorGenerator

class MasterDatasetGenerator:
    def __init__(self, output_dir: str = "dataset", grid_size: int = 128, force_fdm: bool = False):
        self.output_dir = output_dir
        self.grid_size = grid_size
        self.force_fdm = force_fdm
        self.floorplan_gen = FloorplanGenerator()
        self.sensor_gen = SensorGenerator(grid_size=grid_size)

    def prepare_directories(self, splits: List[str] = ["train", "val", "test"]):
        """
        Creates target outputs directory framework conforming to guidelines.
        """
        for split in splits:
            split_path = os.path.join(self.output_dir, split)
            os.makedirs(os.path.join(split_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(split_path, "matrices"), exist_ok=True)
            os.makedirs(os.path.join(split_path, "sensors"), exist_ok=True)
        print(f"[Pipeline] Directory structures prepared in '{self.output_dir}/'", flush=True)

    def generate_single_sample_worker(self, args_tuple: Tuple[int, str, float, float]) -> Dict[str, Any]:
        """
        Multiprocessing worker function generating a single sample and saving its files.
        """
        sample_idx, split, k, boundary_temp = args_tuple
        
        # Instantiate localized solver inside process to prevent state sharing
        solver = ThermalPDESolver(grid_size=self.grid_size)
        
        # 1. Floorplan Generation
        num_blocks = np.random.randint(1, 11)
        blocks = self.floorplan_gen.generate_single_sample(num_blocks=num_blocks)
        
        # 2. PDE Solving
        try:
            temperature_matrix = solver.solve(k=k, ambient=boundary_temp, blocks=blocks, force_fdm=self.force_fdm)
        except Exception as e:
            # Emergency fallback solve
            temperature_matrix = solver.solve_fdm(k=k, ambient=boundary_temp, blocks=blocks)
            
        # 3. Sensor Deployment
        sensor_data_consolidated = self.sensor_gen.process_sample(temperature_matrix)
        
        # Save indices and filenames
        sample_id = f"thermal_{sample_idx:05d}"
        split_path = os.path.join(self.output_dir, split)
        
        # Safe Matrix output (.npy)
        matrix_file = os.path.join(split_path, "matrices", f"{sample_id}.npy")
        np.save(matrix_file, temperature_matrix.astype(np.float32))
        
        # Beautiful Image representation with inferno colormap (.png) - no axes
        img_file = os.path.join(split_path, "images", f"{sample_id}.png")
        fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove padding/borders thoroughly
        ax.axis('off')
        ax.imshow(temperature_matrix, cmap='inferno', origin='lower', aspect='equal')
        fig.savefig(img_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Sensor writing:
        # thermal_00001_sensors.json contains the standard default 5-sensor array
        primary_sensors_file = os.path.join(split_path, "sensors", f"{sample_id}_sensors.json")
        with open(primary_sensors_file, "w") as f:
            json.dump(sensor_data_consolidated["sensors_5"], f, indent=2)
            
        # For researchers' granularity, we write specific ones too
        for density in [5, 8, 10]:
            density_file = os.path.join(split_path, "sensors", f"{sample_id}_sensors_{density}.json")
            with open(density_file, "w") as f:
                json.dump(sensor_data_consolidated[f"sensors_{density}"], f, indent=2)
                
        # Format metadata row properties
        h_positions = [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in blocks]
        h_powers = [b["power_density"] for b in blocks]
        
        metadata_row = {
            "sample_id": sample_id,
            "hotspot_count": len(blocks),
            "hotspot_positions": json.dumps(h_positions),
            "hotspot_power_values": json.dumps(h_powers),
            "thermal_conductivity": k,
            "boundary_temperature": boundary_temp,
            "peak_temperature": float(np.round(temperature_matrix.max(), 4)),
            "min_temperature": float(np.round(temperature_matrix.min(), 4))
        }
        
        return metadata_row

    def execute_split_generation(self, split: str, count: int, start_idx: int) -> pd.DataFrame:
        """
        Coordinates the execution of a single dataset split using multiprocessing.
        """
        print(f"\n[Pipeline] Generating {count} samples for split: '{split}' (starting from ID: {start_idx})...", flush=True)
        
        # Generate random physical physics parameters for each system sample
        # We sample k from uniform [100, 150] and boundary_temp from uniform [20, 40]
        k_values = np.random.uniform(100.0, 150.0, count)
        boundary_temps = np.random.uniform(20.0, 40.0, count)
        
        workers_args = []
        for i in range(count):
            sample_idx = start_idx + i
            workers_args.append((sample_idx, split, k_values[i], boundary_temps[i]))
            
        # Determine pool size
        cpu_count = max(1, mp.cpu_count() - 1)
        print(f"[Pipeline] Spawning ProcessPool executing on {cpu_count} CPU cores...", flush=True)
        
        metadata_rows = []
        with mp.Pool(processes=cpu_count) as pool:
            # We map the task and display simple batch status updates
            results = pool.imap_unordered(self.generate_single_sample_worker, workers_args)
            
            for current_idx, row in enumerate(results):
                metadata_rows.append(row)
                if (current_idx + 1) % max(1, count // 10) == 0 or (current_idx + 1) == count:
                    completed_pct = int(((current_idx + 1) / count) * 100)
                    print(f"[{split.upper()}] Generated {current_idx + 1}/{count} samples ({completed_pct}% completed).", flush=True)
                    
        # Sort metadata by sample_id for sequential alignment
        df_metadata = pd.DataFrame(metadata_rows).sort_values("sample_id").reset_index(drop=True)
        
        # Save CSV inside split directory
        csv_file = os.path.join(self.output_dir, split, "metadata.csv")
        df_metadata.to_csv(csv_file, index=False)
        print(f"[Pipeline] Saved compiled index CSV metadata to {csv_file}", flush=True)
        
        return df_metadata

    def run_pipeline(self, train_cnt: int, val_cnt: int, test_cnt: int):
        """
        Starts the full generation loop across all three data partitions.
        """
        self.prepare_directories()
        
        # Sequences of ID numbers are consecutive across splits to prevent indexing overlaps
        df_train = self.execute_split_generation("train", train_cnt, start_idx=1)
        df_val = self.execute_split_generation("val", val_cnt, start_idx=train_cnt + 1)
        df_test = self.execute_split_generation("test", test_cnt, start_idx=train_cnt + val_cnt + 1)
        
        print("\n" + "="*50)
        print("PINN THERMAL DATASET GENERATION COMPLETE")
        print(f"Total samples produced: {len(df_train) + len(df_val) + len(df_test)}")
        print(f" - Train Split: {len(df_train)} samples")
        print(f" - Validation Split: {len(df_val)} samples")
        print(f" - Test Split: {len(df_test)} samples")
        print("="*50, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN Thermal Dataset Generation Master Orchestrator")
    parser.add_argument("--scale", type=str, choices=["prototype", "production"], default="prototype",
                        help="Pre-configured scale option ('prototype': 500/100/100, 'production': 50000/5000/5000)")
    parser.add_argument("--train-size", type=int, default=None, help="Explicit size for training split")
    parser.add_argument("--val-size", type=int, default=None, help="Explicit size for validation split")
    parser.add_argument("--test-size", type=int, default=None, help="Explicit size for test split")
    parser.add_argument("--grid-size", type=int, default=128, help="Mesh discretization dimensions (Default 128x128)")
    parser.add_argument("--out-dir", type=str, default="dataset", help="Target output folder")
    parser.add_argument("--force-fdm", action="store_true", help="Force fast SciPy FDM solver solver execution")
    
    args = parser.parse_args()
    
    # Establish counts based on scale or manual overrides
    if args.scale == "production":
        train_count = 50000
        val_count = 5000
        test_count = 5000
    else: # prototype split by default
        train_count = 500
        val_count = 100
        test_count = 100
        
    # Manual overrides take precedent if declared
    if args.train_size is not None:
        train_count = args.train_size
    if args.val_size is not None:
        val_count = args.val_size
    if args.test_size is not None:
        test_count = args.test_size
        
    generator = MasterDatasetGenerator(
        output_dir=args.out_dir,
        grid_size=args.grid_size,
        force_fdm=args.force_fdm
    )
    
    generator.run_pipeline(
        train_cnt=train_count,
        val_cnt=val_count,
        test_cnt=test_count
    )
