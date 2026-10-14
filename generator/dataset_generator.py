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
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

from floorplan_generator import FloorplanGenerator, DIE_LX, DIE_LY
from pde_solver import (
    DIE_LX_DEFAULT,
    DIE_LY_DEFAULT,
    H_EFF_DEFAULT,
    FineMeshReferenceSolver,
    ThermalPDESolver,
    T_AMB_MIN_C,
    T_AMB_MAX_C,
    leakage_power_density,
)
from sensor_generator import SensorGenerator

# Maximum operating junction temperature
# [ASSUMPTION — NOT SOURCED: 125 °C is a widely-quoted JEDEC maximum junction
#  temperature, but the specific JEDEC document was not uploaded. No uploaded
#  reference paper cites this value explicitly.]
T_JUNCTION_MAX_C = 125.0   # °C
T_VIOLATION_C    = 125.0   # °C  alias used in violation reporting
T_SILICON_MELT_C = 1412.0  # °C  (silicon melting point, well-known physical constant)
SENSOR_N_SIGMA = 4.0


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

    def execute_split_generation(self, split: str, sample_indices: List[int],
                                  fp_id_offset: int,
                                  save_metadata: bool = True) -> pd.DataFrame:
        count = len(sample_indices)
        print(f"\n[Pipeline] Generating {count} samples for '{split}' "
              f"(IDs {min(sample_indices)}–{max(sample_indices)})...", flush=True)

        T_amb_values = np.random.uniform(T_AMB_MIN_C, T_AMB_MAX_C, count)
        args_list = [
            (self, sample_idx, split, float(T_amb_values[i]), fp_id_offset + i)
            for i, sample_idx in enumerate(sample_indices)
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
        existing_indices = set()
        for sample_id in existing_metadata.get("sample_id", []):
            match = re.fullmatch(r"thermal_(\d+)", str(sample_id))
            if not match:
                raise ValueError(
                    f"Cannot append: invalid sample ID in metadata: {sample_id}"
                )
            existing_indices.add(int(match.group(1)))

        disk_indices = set()
        for root, _, filenames in os.walk(self.output_dir):
            for filename in filenames:
                match = re.search(r"thermal_(\d+)", filename)
                if match:
                    disk_indices.add(int(match.group(1)))

        orphan_indices = sorted(disk_indices - existing_indices)
        if orphan_indices:
            raise ValueError(
                "Cannot append: sample files have no matching metadata rows "
                f"(IDs {orphan_indices[:10]}). Restore their metadata or move "
                "the incomplete files out of the output directory before retrying."
            )
        return max(existing_indices, default=0) + 1

    def _replacement_failures(self, metadata: Dict[str, Any],
                              candidate_dir: str) -> List[str]:
        split = str(metadata["split"])
        sample_id = str(metadata["sample_id"])
        split_dir = os.path.join(candidate_dir, split)
        matrix_path = os.path.join(split_dir, "matrices", f"{sample_id}.npy")
        T = np.load(matrix_path).astype(np.float64)
        failures = []

        if not np.isfinite(T).all():
            return ["temperature_matrix_has_nan_or_inf"]

        T_amb = float(metadata["T_amb_C"])
        T_min = float(T.min())
        T_peak = float(T.max())
        if T_min < T_amb - 0.5:
            failures.append("below_ambient")
        if T_peak > T_JUNCTION_MAX_C:
            failures.append("above_125C")
        if T_peak > T_SILICON_MELT_C:
            failures.append("above_silicon_melt")
        if T_peak - T_min < 0.5:
            failures.append("flat_temperature_field")

        blocks = json.loads(metadata["blocks_json"])
        N_mat = T.shape[0]
        solver = ThermalPDESolver(
            grid_size=N_mat,
            L_x=DIE_LX_DEFAULT,
            L_y=DIE_LY_DEFAULT,
            h_eff=H_EFF_DEFAULT,
        )
        Q = solver._build_Q_map(blocks, T)
        P_in = float(np.sum(Q) * solver.dx * solver.dy)
        P_out = float(np.sum(H_EFF_DEFAULT * (T - T_amb)) * solver.dx * solver.dy)
        denom = max(abs(P_in), abs(P_out), 1e-12)
        energy_error = abs(P_in - P_out) / denom * 100.0
        if not np.isfinite(energy_error):
            failures.append("non_finite_energy_balance")

        j_idx = np.arange(N_mat)
        x_left = (j_idx - 0.5) * solver.dx
        x_right = (j_idx + 0.5) * solver.dx
        i_idx = np.arange(N_mat)
        y_bottom = (i_idx - 0.5) * solver.dy
        y_top = (i_idx + 0.5) * solver.dy
        analytic_power = 0.0
        for block in blocks:
            area = (block["x_max"] - block["x_min"]) * (
                block["y_max"] - block["y_min"]
            )
            overlap_x = np.maximum(
                0.0,
                np.minimum(block["x_max"], x_right)
                - np.maximum(block["x_min"], x_left),
            )
            overlap_y = np.maximum(
                0.0,
                np.minimum(block["y_max"], y_top)
                - np.maximum(block["y_min"], y_bottom),
            )
            fractions = np.outer(overlap_y / solver.dy, overlap_x / solver.dx)
            fraction_sum = np.sum(fractions)
            mean_temp = (
                float(np.sum(fractions * T) / fraction_sum)
                if fraction_sum > 0 else T_amb
            )
            leakage = leakage_power_density(
                np.array([mean_temp]),
                float(block.get("power_density_leakage0", 0.0)),
                float(block.get("leakage_beta", 0.012)),
            )[0]
            analytic_power += (
                float(block.get("power_density_dynamic", 0.0)) + leakage
            ) * 1e4 * area

        relative_power_error = (
            abs(P_in - analytic_power) / max(analytic_power, 1e-12) * 100.0
        )
        if relative_power_error > 0.5:
            failures.append(f"analytic_power_{relative_power_error:.4f}%")

        for density in (5, 8, 10):
            sensor_path = os.path.join(
                split_dir, "sensors", f"{sample_id}_sensors_{density}.json"
            )
            with open(sensor_path, encoding="utf-8") as sensor_file:
                sensor_data = json.load(sensor_file)
            sigma = float(sensor_data.get("noise_sigma_C", 1.0))
            quant = float(sensor_data.get("quant_step_C", 1.0))
            true_temps = sensor_data.get("temperatures_true", [])
            noisy_temps = sensor_data.get("temperatures_noisy", [])
            offsets = sensor_data.get(
                "calibration_offsets_C", [0.0] * len(true_temps)
            )
            tolerance = SENSOR_N_SIGMA * sigma + quant
            if not (
                len(true_temps) == len(noisy_temps) == len(offsets)
                and all(
                    abs((true_temp + offset) - noisy_temp) <= tolerance
                    for true_temp, noisy_temp, offset
                    in zip(true_temps, noisy_temps, offsets)
                )
            ):
                failures.append(f"sensor_noise_{density}")
                break

        return failures

    @staticmethod
    def _write_csv_atomically(dataframe: pd.DataFrame, path: str) -> None:
        temp_path = f"{path}.tmp"
        dataframe.to_csv(temp_path, index=False)
        os.replace(temp_path, path)

    def regenerate_violation_samples(self, max_attempts_per_sample: int = 50) -> pd.DataFrame:
        if max_attempts_per_sample < 1:
            raise ValueError("max_attempts_per_sample must be at least 1")
        metadata = self._load_existing_metadata()
        if metadata.empty:
            raise ValueError("No metadata found to regenerate. Please generate a dataset first.")
        violation_mask = metadata["thermal_violation"].astype(str).str.lower().eq("true")
        violations = metadata.loc[violation_mask].copy()

        if violations.empty:
            print("[Pipeline] No thermal-violation samples found; nothing to regenerate.")
            return metadata

        print(
            f"[Pipeline] Replacing {len(violations)} thermal-violation sample(s); "
            f"max attempts per sample: {max_attempts_per_sample}.",
            flush=True,
        )
        os.makedirs(self.output_dir, exist_ok=True)
        next_floorplan_id = int(metadata["floorplan_id"].max()) + 1

        for _, row in violations.sort_values("sample_id").iterrows():
            sample_id = str(row["sample_id"])
            sample_idx = int(sample_id.split("_")[-1])
            split = str(row["split"])
            rng = np.random.default_rng(self.global_seed + sample_idx * 1_000_003)
            replacement = None
            last_failures = []

            for attempt in range(1, max_attempts_per_sample + 1):
                T_amb = float(rng.uniform(T_AMB_MIN_C, T_AMB_MAX_C))
                floorplan_id = next_floorplan_id
                next_floorplan_id += 1

                with tempfile.TemporaryDirectory(
                    prefix=f"{sample_id}_attempt_{attempt}_",
                    dir=self.output_dir,
                ) as candidate_dir:
                    previous_output_dir = self.output_dir
                    self.output_dir = candidate_dir
                    self.prepare_directories([split])
                    try:
                        generated = self.generate_single_sample(
                            sample_idx, split, T_amb, floorplan_id
                        )
                    finally:
                        self.output_dir = previous_output_dir

                    last_failures = self._replacement_failures(generated, candidate_dir)
                    if not last_failures:
                        replacement = generated
                        for subdir, filenames in (
                            ("matrices", [f"{sample_id}.npy", f"{sample_id}_Q.npy"]),
                            ("images", [f"{sample_id}.png"]),
                            ("sensors", [
                                f"{sample_id}_sensors_{density}.json"
                                for density in (5, 8, 10)
                            ]),
                        ):
                            source_dir = os.path.join(candidate_dir, split, subdir)
                            target_dir = os.path.join(self.output_dir, split, subdir)
                            os.makedirs(target_dir, exist_ok=True)
                            for filename in filenames:
                                source = os.path.join(source_dir, filename)
                                if os.path.isfile(source):
                                    shutil.copyfile(
                                        source, os.path.join(target_dir, filename)
                                    )
                        break

                print(
                    f"  {sample_id}: attempt {attempt}/{max_attempts_per_sample} "
                    f"failed checks: {', '.join(last_failures)}",
                    flush=True,
                )

            if replacement is None:
                raise RuntimeError(
                    f"Could not produce a valid replacement for {sample_id} "
                    f"after {max_attempts_per_sample} attempts; last failures: "
                    f"{', '.join(last_failures)}. Existing metadata was left unchanged "
                    "for this sample; rerun with a larger --max-attempts-per-sample."
                )

            for column, value in replacement.items():
                if column in metadata.columns:
                    metadata.loc[metadata["sample_id"] == sample_id, column] = value
            if "peak_C" in metadata.columns:
                metadata.loc[metadata["sample_id"] == sample_id, "peak_C"] = replacement["peak_temp_C"]
            if "violates_125C" in metadata.columns:
                metadata.loc[metadata["sample_id"] == sample_id, "violates_125C"] = False

            metadata = metadata.sort_values("sample_id").reset_index(drop=True)
            for split_name in ("train", "validation", "test"):
                split_df = metadata[metadata["split"] == split_name].copy()
                split_metadata_path = os.path.join(
                    self.output_dir, split_name, "metadata.csv"
                )
                self._write_csv_atomically(split_df, split_metadata_path)
            self._write_csv_atomically(
                metadata, os.path.join(self.output_dir, "metadata.csv")
            )
            self._write_csv_atomically(
                metadata[["sample_id", "split"]],
                os.path.join(self.output_dir, "splits.csv"),
            )
            print(
                f"  Replaced {sample_id} in {split}: peak "
                f"{replacement['peak_temp_C']:.2f}°C "
                f"(attempts used: {attempt})",
                flush=True,
            )

        remaining_violations = metadata[
            metadata["thermal_violation"].astype(str).str.lower().eq("true")
        ]
        if not remaining_violations.empty:
            raise RuntimeError(
                "Regeneration ended with thermal violations: "
                + ", ".join(remaining_violations["sample_id"].tolist())
            )
        print(
            f"[Pipeline] Replaced {len(violations)} sample(s); all now satisfy "
            "the 125°C limit and replacement checks.",
            flush=True,
        )
        return metadata

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

        split_counts = {
            "train": train_cnt,
            "validation": val_cnt,
            "test": test_cnt,
        }
        sample_indices_by_split = {split: [] for split in split_counts}
        if append:
            # Keep the requested counts while distributing each split across
            # the consecutive new sample IDs, as in a randomized dataset split.
            assignments = [
                split for split, count in split_counts.items()
                for _ in range(count)
            ]
            np.random.default_rng(self.global_seed).shuffle(assignments)
            for offset, split in enumerate(assignments):
                sample_indices_by_split[split].append(sample_start + offset)
        else:
            next_idx = sample_start
            for split, count in split_counts.items():
                sample_indices_by_split[split] = list(range(next_idx, next_idx + count))
                next_idx += count

        # Floorplan IDs remain globally unique across splits (no leakage).
        floorplan_offsets = {
            "train": floorplan_start,
            "validation": floorplan_start + train_cnt,
            "test": floorplan_start + train_cnt + val_cnt,
        }
        generated_by_split = {}
        for split in ("train", "validation", "test"):
            generated_by_split[split] = self.execute_split_generation(
                split,
                sample_indices_by_split[split],
                fp_id_offset=floorplan_offsets[split],
                save_metadata=not append,
            )

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
              f"(train={train_cnt}, validation={val_cnt}, test={test_cnt} added)")
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
    parser.add_argument("--regenerate-violations", action="store_true",
                        help="Replace thermal violations, retrying until the 125C limit and dataset checks pass")
    parser.add_argument("--max-attempts-per-sample", type=int, default=50,
                        help="Maximum replacement attempts before stopping (default: 50)")
    parser.add_argument("--no-fine-mesh", action="store_true",
                        help="Use coarse 128x128 solver (disables anti-inverse-crime)")
    args = parser.parse_args()

    if args.append and args.regenerate_violations:
        raise ValueError("Use either --append or --regenerate-violations, not both.")

    if args.regenerate_violations:
        gen = MasterDatasetGenerator(
            output_dir=args.out_dir,
            grid_size=args.grid_size,
            use_fine_mesh=not args.no_fine_mesh,
            global_seed=args.seed,
        )
        gen.regenerate_violation_samples(
            max_attempts_per_sample=args.max_attempts_per_sample
        )
        raise SystemExit(0)

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
