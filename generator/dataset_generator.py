#!/usr/bin/env python3
"""
Orchestrated Dataset Generator (v2 – peer-review revision)
Part of the PINN Thermal Dataset Generation Pipeline.

Key changes from v1:
  - All units SI (metres, W/cm², °C).
  - Ground truth generated with FineMeshReferenceSolver (512x512) to avoid
    the inverse crime (Wirgin, arXiv math-ph/0401050).
  - Coarse solver (128x128) tag recorded in metadata separately.
  - No overlapping blocks, no summed power densities.
  - Sensor positions fixed at design time (block centres).
  - Noisy sensor readings with explicit noise model.
  - metadata.csv includes solver_tag, energy_bal_err_pct, thermo_iters.
  - Thermal violation samples flagged but NOT dropped.
  - Ambient temperature sampled from 25-45 °C.

Anti-inverse-crime (Wirgin arXiv math-ph/0401050):
  Ground-truth .npy files are generated with FineMeshReferenceSolver
  (solver_tag = 'FineMesh512_to_128', fine mesh = 512×512, then
  downsampled to 128×128 by stride [::4, ::4]). Reconstruction models
  should use the coarse solver (128×128) for forward modeling.
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

from floorplan_generator import FloorplanGenerator, DIE_LX, DIE_LY
from pde_solver import FineMeshReferenceSolver, ThermalPDESolver, T_AMB_MIN_C, T_AMB_MAX_C
from sensor_generator import SensorGenerator

# Maximum operating junction temperature
# [ASSUMPTION — NOT SOURCED: 125 °C is a widely-quoted JEDEC maximum junction
#  temperature, but the specific JEDEC document was not uploaded. No uploaded
#  reference paper cites this value explicitly.]
T_JUNCTION_MAX_C = 125.0   # °C
T_VIOLATION_C    = 125.0   # °C  alias used in violation reporting
T_SILICON_MELT_C = 1412.0  # °C  (silicon melting point, well-known physical constant)


def _worker_generate_sample(args):
    generator_inst, sample_idx, split, T_amb, floorplan_id = args
    return generator_inst.generate_single_sample(sample_idx, split, T_amb, floorplan_id)


class MasterDatasetGenerator:
    def __init__(self, output_dir: str = "dataset",
                 grid_size: int = 128,
                 use_fine_mesh: bool = True,
                 global_seed: int = 42):
        self.output_dir = output_dir
        self.grid_size = grid_size
        self.use_fine_mesh = use_fine_mesh
        self.global_seed = global_seed
        np.random.seed(global_seed)

        # One floorplan generator with reproducible seed
        self.floorplan_gen = FloorplanGenerator(L_x=DIE_LX, L_y=DIE_LY,
                                                rng_seed=global_seed)
        # Sensor generator
        self.sensor_gen = SensorGenerator(grid_size=grid_size,
                                          L_x=DIE_LX, L_y=DIE_LY,
                                          rng_seed=global_seed + 1)

    def prepare_directories(self, splits: List[str] = ["train", "validation", "test"]):
        for split in splits:
            sp = os.path.join(self.output_dir, split)
            os.makedirs(os.path.join(sp, "images"),   exist_ok=True)
            os.makedirs(os.path.join(sp, "matrices"), exist_ok=True)
            os.makedirs(os.path.join(sp, "sensors"),  exist_ok=True)
        print(f"[Pipeline] Directories prepared in '{self.output_dir}/'", flush=True)

    def generate_single_sample(self, sample_idx: int, split: str,
                                T_amb: float,
                                floorplan_id: int) -> Dict[str, Any]:
        """
        Generates one sample. Ground truth uses FineMeshReferenceSolver (512x512 downsampled to 128x128).
        """
        # ----- Floorplan -------------------------------------------------
        # Each floorplan_id maps to a unique floorplan (train/val/test split
        # by floorplan_id to avoid data leakage across splits).
        fp_rng = np.random.default_rng(self.global_seed + floorplan_id * 100)
        fp_gen = FloorplanGenerator(L_x=DIE_LX, L_y=DIE_LY,
                                    rng_seed=self.global_seed + floorplan_id * 100)
        blocks = fp_gen.generate_single_sample(grid_size=self.grid_size)

        if len(blocks) == 0:
            # Degenerate case – shouldn't happen but guard anyway
            blocks = fp_gen.generate_single_sample(2, grid_size=self.grid_size)

        # ----- Solve (fine mesh → 128×128 output) ------------------------
        if self.use_fine_mesh:
            solver = FineMeshReferenceSolver(base_grid_size=self.grid_size, verbose=False)
        else:
            solver = ThermalPDESolver(grid_size=self.grid_size, verbose=False)

        result = (solver.solve_and_downsample(T_amb, blocks)
                  if self.use_fine_mesh
                  else solver.solve(T_amb, blocks))

        T_matrix = result["T"]          # (128, 128) float32 °C
        Q_matrix = result.get("Q_total", None)

        # ----- Sensor placement ------------------------------------------
        s_rng = np.random.default_rng(self.global_seed + sample_idx * 7)
        s_gen = SensorGenerator(grid_size=self.grid_size, L_x=DIE_LX, L_y=DIE_LY,
                                 rng_seed=self.global_seed + sample_idx * 7)
        sensor_data = s_gen.process_sample(T_matrix, blocks)

        # ----- File IDs --------------------------------------------------
        sample_id = f"thermal_{sample_idx:05d}"
        split_path = os.path.join(self.output_dir, split)

        # ----- Save .npy -------------------------------------------------
        mat_file = os.path.join(split_path, "matrices", f"{sample_id}.npy")
        np.save(mat_file, T_matrix)

        # Save Q map alongside for energy-balance validation
        if Q_matrix is not None:
            q_file = os.path.join(split_path, "matrices", f"{sample_id}_Q.npy")
            np.save(q_file, Q_matrix.astype(np.float32))

        # ----- Save PNG heatmap ------------------------------------------
        img_file = os.path.join(split_path, "images", f"{sample_id}.png")
        fig, ax = plt.subplots(figsize=(4, 4), dpi=128)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis("off")
        ax.imshow(T_matrix, cmap="inferno", origin="lower", aspect="equal")
        fig.savefig(img_file, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # ----- Save sensors (three counts, no default duplicate) ----------
        for density in [5, 8, 10]:
            df_file = os.path.join(split_path, "sensors",
                                   f"{sample_id}_sensors_{density}.json")
            with open(df_file, "w") as f:
                json.dump(sensor_data[f"sensors_{density}"], f, indent=2)

        # ----- Thermal violation flag ------------------------------------
        peak_T = float(T_matrix.max())
        is_violation = bool(peak_T > T_JUNCTION_MAX_C)

        # ----- Metadata row ----------------------------------------------
        blocks_json = FloorplanGenerator.blocks_to_json_safe(blocks)
        solver_tag_val = "FineMesh512_to_128" if self.use_fine_mesh else result.get("solver_tag", "unknown")
        meta = {
            "sample_id":          sample_id,
            "split":              split,
            "floorplan_id":       floorplan_id,
            "solver_tag":         solver_tag_val,
            "T_amb_C":            float(T_amb),
            "peak_temp_C":        float(np.round(peak_T, 4)),
            "min_temp_C":         float(np.round(T_matrix.min(), 4)),
            "thermal_violation":  is_violation,
            "energy_bal_P_in":    float(np.round(result["energy_bal"]["P_in_W"], 4)),
            "energy_bal_P_out":   float(np.round(result["energy_bal"]["P_out_W"], 4)),
            "energy_bal_err_pct": float(np.round(result["energy_bal"]["error_pct"], 3)),
            # Note: newton_iters records only the final outer-loop inner Picard iteration count
            "newton_iters":       result.get("newton_iters", -1),
            "thermo_iters":       result.get("thermo_iters", -1),
            "block_count":        len(blocks),
            # Serialise blocks as JSON strings (SI units)
            "blocks_json":        json.dumps(blocks_json),
            # Physical parameter metadata
            "die_L_x_m":          DIE_LX,
            "die_L_y_m":          DIE_LY,
            "units":              "coords_m_power_W_per_cm2_temp_C",
        }
        return meta

    def execute_split_generation(self, split: str, count: int,
                                  start_idx: int, fp_id_offset: int,
                                  save_metadata: bool = True) -> pd.DataFrame:
        print(f"\n[Pipeline] Generating {count} samples for '{split}' "
              f"(IDs {start_idx}–{start_idx+count-1})...", flush=True)

        T_amb_values = np.random.uniform(T_AMB_MIN_C, T_AMB_MAX_C, count)
        args_list = [
            (self, start_idx + i, split, float(T_amb_values[i]), fp_id_offset + i)
            for i in range(count)
        ]

        import multiprocessing as mp
        cpu_count = max(1, min(count, mp.cpu_count() - 1, 4))
        print(f"  [Pipeline] Running with ProcessPool on {cpu_count} CPU cores...", flush=True)

        t0 = time.time()
        rows = []
        with mp.Pool(processes=cpu_count) as pool:
            for i, row in enumerate(pool.imap_unordered(_worker_generate_sample, args_list)):
                rows.append(row)
                if (i + 1) % max(1, count // 10) == 0 or (i + 1) == count:
                    elapsed = time.time() - t0
                    pct = (i + 1) / count * 100
                    print(f"  [{split.upper()}] {i+1}/{count} ({pct:.0f}%)  elapsed {elapsed:.1f}s", flush=True)

        df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
        csv_file = os.path.join(self.output_dir, split, "metadata.csv")
        if save_metadata:
            df.to_csv(csv_file, index=False)
        n_viol = int(df["thermal_violation"].sum()) if "thermal_violation" in df else 0
        print(
            f"  Generated {len(df)} rows"
            + (f" → {csv_file}" if save_metadata else "")
            + f" (thermal violations: {n_viol})",
            flush=True,
        )
        return df

    def _load_existing_metadata(self) -> pd.DataFrame:
        split_metadata = []
        for split in ["train", "validation", "test"]:
            csv_file = os.path.join(self.output_dir, split, "metadata.csv")
            if os.path.isfile(csv_file):
                split_metadata.append(pd.read_csv(csv_file))

        if split_metadata:
            existing = pd.concat(split_metadata, ignore_index=True)
        else:
            csv_file = os.path.join(self.output_dir, "metadata.csv")
            if not os.path.isfile(csv_file):
                return pd.DataFrame()
            existing = pd.read_csv(csv_file)

        required = {"sample_id", "split", "floorplan_id"}
        missing = required.difference(existing.columns)
        if missing:
            raise ValueError(
                f"Cannot append: existing metadata is missing columns {sorted(missing)}"
            )
        if existing["sample_id"].duplicated().any():
            duplicates = existing.loc[
                existing["sample_id"].duplicated(), "sample_id"
            ].tolist()
            raise ValueError(
                f"Cannot append: duplicate sample IDs in existing metadata: {duplicates}"
            )
        return existing

    def _next_sample_index(self, existing_metadata: pd.DataFrame) -> int:
        max_index = 0
        for sample_id in existing_metadata.get("sample_id", []):
            match = re.fullmatch(r"thermal_(\d+)", str(sample_id))
            if not match:
                raise ValueError(
                    f"Cannot append: invalid sample ID in metadata: {sample_id}"
                )
            max_index = max(max_index, int(match.group(1)))

        for root, _, filenames in os.walk(self.output_dir):
            for filename in filenames:
                match = re.search(r"thermal_(\d+)", filename)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        return max_index + 1

    def run_pipeline(self, train_cnt: int, val_cnt: int, test_cnt: int,
                     append: bool = False):
        existing_metadata = self._load_existing_metadata() if append else pd.DataFrame()
        self.prepare_directories()

        if append:
            sample_start = self._next_sample_index(existing_metadata)
            floorplan_start = (
                int(existing_metadata["floorplan_id"].max()) + 1
                if not existing_metadata.empty else 0
            )
        else:
            sample_start = 1
            floorplan_start = 0

        # Floorplan IDs are globally unique across splits (no leakage).
        df_train = self.execute_split_generation("train", train_cnt,
                                                  start_idx=sample_start,
                                                  fp_id_offset=floorplan_start,
                                                  save_metadata=not append)
        df_val   = self.execute_split_generation("validation", val_cnt,
                                                  start_idx=sample_start + train_cnt,
                                                  fp_id_offset=floorplan_start + train_cnt,
                                                  save_metadata=not append)
        df_test  = self.execute_split_generation("test",  test_cnt,
                                                  start_idx=sample_start + train_cnt + val_cnt,
                                                  fp_id_offset=floorplan_start + train_cnt + val_cnt,
                                                  save_metadata=not append)

        generated_by_split = {
            "train": df_train,
            "validation": df_val,
            "test": df_test,
        }
        if append:
            for split, generated in generated_by_split.items():
                prior = existing_metadata[existing_metadata["split"] == split]
                combined = pd.concat([prior, generated], ignore_index=True)
                combined = combined.sort_values("sample_id").reset_index(drop=True)
                combined.to_csv(
                    os.path.join(self.output_dir, split, "metadata.csv"), index=False
                )
                generated_by_split[split] = combined

        total_df = pd.concat(list(generated_by_split.values()), ignore_index=True)
        total_df = total_df.sort_values("sample_id").reset_index(drop=True)
        total_df.to_csv(os.path.join(self.output_dir, "metadata.csv"), index=False)
        total_df[["sample_id", "split"]].to_csv(
            os.path.join(self.output_dir, "splits.csv"), index=False
        )

        total = len(total_df)
        print(f"\n{'='*60}")
        print(f"PINN THERMAL DATASET GENERATION COMPLETE")
        print(f"Total samples: {total}  "
              f"(train={len(df_train)}, validation={len(df_val)}, test={len(df_test)})")
        print(f"Global seed: {self.global_seed}")
        print(f"Solver: {'FineMesh512_to_128' if self.use_fine_mesh else 'Coarse-128'}")
        print(f"Anti-inverse-crime: {'YES (Wirgin 2001)' if self.use_fine_mesh else 'NO'}")
        print("="*60, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PINN Thermal Dataset Generator v2 (peer-review revision)")
    parser.add_argument("--scale", choices=["prototype", "production"],
                        default="prototype",
                        help="prototype=500/100/100, production=50000/5000/5000")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size",   type=int, default=None)
    parser.add_argument("--test-size",  type=int, default=None)
    parser.add_argument("--grid-size",  type=int, default=128)
    parser.add_argument("--out-dir",    type=str, default="dataset")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Global random seed for reproducibility")
    parser.add_argument("--append", action="store_true",
                        help="Append samples and continue IDs after existing output")
    parser.add_argument("--no-fine-mesh", action="store_true",
                        help="Use coarse 128x128 solver (disables anti-inverse-crime)")
    args = parser.parse_args()

    if args.scale == "production":
        train_count, val_count, test_count = 50000, 5000, 5000
    else:
        train_count, val_count, test_count = 500, 100, 100

    if args.train_size is not None: train_count = args.train_size
    if args.val_size   is not None: val_count   = args.val_size
    if args.test_size  is not None: test_count  = args.test_size

    gen = MasterDatasetGenerator(
        output_dir=args.out_dir,
        grid_size=args.grid_size,
        use_fine_mesh=not args.no_fine_mesh,
        global_seed=args.seed,
    )
    gen.run_pipeline(train_count, val_count, test_count, append=args.append)
