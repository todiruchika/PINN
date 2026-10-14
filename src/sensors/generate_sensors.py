"""Generate sensor JSON files for all samples using the POD-QR layout.

Writes data/generated/<split>/thermal_NNNNN_sensors_N.json for N in {5,8,10}.
Output is deterministic given the global seed and sample ID.
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from src.data.splits import load_splits
from src.data.dataset import load_metadata, RAW_DIR
from src.sensors.layouts import podqr_layout
from src.sensors.readers import idx_to_metre

SENSOR_COUNTS = [5, 8, 10]
NOISE_SIGMA = 1.0
QUANT_STEP = 1.0
CALIB_SIGMA = 0.5
GLOBAL_SEED = 42
DIE_L = 0.012
GRID = 128
MARGIN = 3


def _calib_offsets(n: int, global_seed: int) -> list:
    """Per-sensor-index calibration offsets (indices 0..n-1), seeded globally."""
    rng = np.random.default_rng(global_seed + 1000)
    offsets = rng.normal(0, CALIB_SIGMA, size=n)
    return [float(offsets[i]) for i in range(n)]


def sensor_record_for_layout(
    T: np.ndarray,
    layout: list,
    n_sensors: int,
    placement_strategy: str,
    sample_num: int,
    global_seed: int = GLOBAL_SEED,
) -> dict:
    positions_m = [[idx_to_metre(col), idx_to_metre(row)] for row, col in layout]
    temperatures_true = [float(T[row, col]) for row, col in layout]
    calibration_offsets = _calib_offsets(n_sensors, global_seed)
    rng = np.random.default_rng(global_seed * 10000 + sample_num * 100 + n_sensors)
    temperatures_noisy = [
        float(round((temperature + calibration_offsets[i] + rng.normal(0, NOISE_SIGMA)) / QUANT_STEP) * QUANT_STEP)
        for i, temperature in enumerate(temperatures_true)
    ]
    return {
        "sensor_count": n_sensors,
        "positions": positions_m,
        "temperatures_true": temperatures_true,
        "temperatures_noisy": temperatures_noisy,
        "noise_sigma_C": NOISE_SIGMA,
        "quant_step_C": QUANT_STEP,
        "calibration_offsets_C": calibration_offsets,
        "placement_strategy": placement_strategy,
        "boundary_margin_cells": MARGIN,
    }


def generate_sensors(
    raw_dir: Path = RAW_DIR,
    out_base: Path = Path("data/generated"),
    global_seed: int = GLOBAL_SEED,
) -> None:
    splits = load_splits(raw_dir / "splits.csv")

    for split, ids in splits.items():
        out_dir = out_base / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for sid in sorted(ids):
            T = np.load(str(raw_dir / split / "matrices" / f"{sid}.npy"))
            sample_num = int(sid.split("_")[1])

            for n in SENSOR_COUNTS:
                layout = podqr_layout(n)
                record = sensor_record_for_layout(
                    T, layout, n, "podqr", sample_num, global_seed
                )
                out_path = out_dir / f"{sid}_sensors_{n}.json"
                with open(out_path, "w") as f:
                    json.dump(record, f, indent=2)

    total = sum(len(v) for v in splits.values()) * len(SENSOR_COUNTS)
    print(f"Generated {total} sensor files in {out_base}/")


if __name__ == "__main__":
    generate_sensors()
