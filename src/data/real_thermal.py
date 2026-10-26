"""Real infrared-camera thermal maps (Lu and Tan, MLCAD 2024) adapted to the sensor-reconstruction task.

Source: https://github.com/sheldonucr/commercial_thermal_map_dataset  (public sample, 100 frames per file).
Only the thermal maps ("output") are used. The performance-counter inputs are ignored, because this project
reconstructs a map from a few temperature sensors.

Adaptation (all choices are assumptions, listed in results/real_data/README.md):
  - every frame is resampled to 128 x 128 (bilinear), which squashes the camera's aspect ratio;
  - virtual sensors: POD-QR pivots fitted on the TRAINING frames only (nested 5 < 8 < 10), read from the map
    at grid nodes and corrupted with Gaussian noise (default sigma 1 degC, fixed once per frame);
  - the ambient temperature is not recorded, so a constant 25 degC is used;
  - the split is by WORKLOAD, so test workloads (heat patterns) are never seen in training.

The pickle files are untrusted downloads, so they are read with a restricted unpickler that only allows
NumPy arrays.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import zoom

GRID = 128
DIE_L = 0.012                      # positions are expressed on the same 12 mm scale as the synthetic chips
AMBIENT_C = 25.0
SENSOR_MARGIN = 3
RAW_DIR = Path("data/real_raw")    # clone of the dataset's data_files/ folder (not committed)
CACHE_DIR = Path("data/real_cache")
I7_WORKLOADS = ["aio-stress", "aobench", "build-gcc", "cachebench", "compress-7zip", "cyclictest", "dbench",
                "encode-flac", "fio", "gimp", "git", "idle", "mbw", "phpbench", "ramspeed", "stream",
                "t-test1", "tinymembench", "tiobench"]


class _ArraysOnly(pickle.Unpickler):
    ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
               ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def read_maps(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        return np.asarray(_ArraysOnly(f).load()["output"], dtype=np.float32)


def resample(maps: np.ndarray) -> np.ndarray:
    factors = (1, GRID / maps.shape[1], GRID / maps.shape[2])
    return zoom(maps, factors, order=1).astype(np.float32)


def load_workloads(chip: str = "i7", workloads: list[str] | None = None) -> dict[str, np.ndarray]:
    """Resampled maps (100, 128, 128) per workload, cached as .npz after the first read."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in workloads or I7_WORKLOADS:
        cache = CACHE_DIR / f"{chip}_{name}.npy"
        if cache.exists():
            out[name] = np.load(cache)
            continue
        maps = resample(read_maps(RAW_DIR / f"CPU_{chip}_{name}.pkl"))
        np.save(cache, maps)
        out[name] = maps
    return out


def split_workloads(maps: dict[str, np.ndarray]) -> dict[str, list[str]]:
    """Deterministic, temperature-stratified split: sort by mean temperature, every 6th goes to test or validation."""
    order = sorted(maps, key=lambda n: float(maps[n].mean()))
    split = {"train": [], "validation": [], "test": []}
    for rank, name in enumerate(order):
        split["test" if rank % 6 == 2 else "validation" if rank % 6 == 5 else "train"].append(name)
    return split


def podqr_positions(train_maps: np.ndarray, n_max: int = 10, rank: int = 10) -> np.ndarray:
    """Grid (row, col) of the first n_max POD-QR pivots, fitted on training frames only. Nested by construction."""
    from scipy.linalg import qr
    X = (train_maps - AMBIENT_C).reshape(len(train_maps), -1).T
    X = X - X.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    mask = np.zeros((GRID, GRID), dtype=bool)
    mask[SENSOR_MARGIN:GRID - SENSOR_MARGIN, SENSOR_MARGIN:GRID - SENSOR_MARGIN] = True
    valid = np.where(mask.ravel())[0]
    _, _, pivots = qr(U[valid, :rank].T, pivoting=True)
    rows, cols = np.unravel_index(valid[pivots[:n_max]], (GRID, GRID))
    return np.stack([rows, cols], axis=1)


class RealThermal:
    """Splits, virtual sensors and the data dictionaries the trainers expect."""

    def __init__(self, chip: str = "i7", noise_c: float = 1.0, noise_seed: int = 0) -> None:
        self.maps = load_workloads(chip)
        self.workloads = split_workloads(self.maps)
        cache = CACHE_DIR / f"{chip}_podqr_pivots.npy"          # the SVD is slow; train workloads are fixed, so cache it
        if cache.exists():
            self.pivots = np.load(cache)
        else:
            self.pivots = podqr_positions(np.concatenate([self.maps[n] for n in self.workloads["train"]]))
            np.save(cache, self.pivots)
        self.noise_c, self.noise_seed = noise_c, noise_seed
        self.splits = {k: [(k, n, i) for n in v for i in range(len(self.maps[n]))] for k, v in self.workloads.items()}

    def arrays(self, ids: list, n_sensors: int) -> dict:
        T = np.stack([self.maps[n][i] for _, n, i in ids])
        rows, cols = self.pivots[:n_sensors, 0], self.pivots[:n_sensors, 1]
        clean = T[:, rows, cols]
        rng = np.random.default_rng([self.noise_seed, n_sensors, len(ids)])
        readings = clean + rng.normal(0.0, self.noise_c, clean.shape).astype(np.float32)
        positions = np.stack([cols, rows], axis=1) * DIE_L / (GRID - 1)            # (N, 2) as (x, y) in metres
        return {
            "ids": ids,
            "T": torch.tensor(T),
            "Q": torch.zeros(len(ids), GRID, GRID),                                  # no source is known for real chips
            "T_amb": torch.full((len(ids),), AMBIENT_C),
            "positions": torch.tensor(np.broadcast_to(positions, (len(ids),) + positions.shape).copy(), dtype=torch.float32),
            "readings": torch.tensor(readings, dtype=torch.float32),
        }
