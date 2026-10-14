"""Bulk loading of a whole split into numpy arrays, for models trained across chips.

``src/data/dataset.py`` loads one sample at a time; a network trained on 500 chips
wants every field and every sensor record in memory once. Everything here reads
the same files (``data/raw/dataset`` and ``data/generated``) and applies the same
POD-QR staleness check as ``load_sample``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import GENERATED_SENSOR_DIR, GRID, RAW_DIR
from src.data.splits import load_splits
from src.sensors.layouts import is_current_podqr_record

DIE_L = 0.012


@dataclass
class SensorNoise:
    """The sensor model recorded in the sensor JSON files (``generate_sensors.py``).

    ``noisy = round((true + offset_i + N(0, sigma)) / quant) * quant``, with the
    per-sensor-index offsets shared by every chip.
    """

    sigma_C: float
    quant_step_C: float
    offsets_C: np.ndarray            # (N,)

    def apply(self, true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Fresh noisy readings for ``true`` of shape ``(..., N)``."""
        noisy = true + self.offsets_C + rng.normal(0.0, self.sigma_C, size=true.shape)
        return np.round(noisy / self.quant_step_C) * self.quant_step_C


@dataclass
class SplitArrays:
    """Everything the model needs for one split and one sensor count."""

    split: str
    n_sensors: int
    ids: list
    T_amb: np.ndarray                # (n,) degC, known per chip
    positions_m: np.ndarray          # (N, 2) sensor positions, identical for every chip
    sensors_true: np.ndarray         # (n, N) degC, noise-free readings
    sensors_noisy: np.ndarray        # (n, N) degC, the stored model input
    noise: SensorNoise
    T: np.ndarray | None = field(default=None, repr=False)   # (n, 128, 128) degC, ground truth
    Q: np.ndarray | None = field(default=None, repr=False)   # (n, 128, 128) W/m^2, ground truth

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, count: int) -> "SplitArrays":
        """The first ``count`` chips (deterministic, for quick validation passes)."""
        sl = slice(0, count)
        return SplitArrays(
            split=self.split, n_sensors=self.n_sensors, ids=self.ids[sl], T_amb=self.T_amb[sl],
            positions_m=self.positions_m, sensors_true=self.sensors_true[sl],
            sensors_noisy=self.sensors_noisy[sl], noise=self.noise,
            T=None if self.T is None else self.T[sl], Q=None if self.Q is None else self.Q[sl],
        )


def load_split_arrays(
    split: str,
    n_sensors: int,
    raw_dir: Path = RAW_DIR,
    generated_dir: Path = GENERATED_SENSOR_DIR,
    with_fields: bool = True,
) -> SplitArrays:
    """Load all chips of ``split`` (``train`` / ``validation`` / ``test``)."""
    if split == "val":
        split = "validation"
    ids = load_splits(raw_dir / "splits.csv")[split]
    if not ids:
        raise ValueError(f"no chips in split {split!r}")

    meta = pd.read_csv(raw_dir / "metadata.csv").set_index("sample_id")
    T_amb = meta.loc[ids, "T_amb_C"].to_numpy(dtype=np.float64, copy=True)

    true_rows, noisy_rows = [], []
    positions = None
    noise = None
    for sid in ids:
        path = generated_dir / split / f"{sid}_sensors_{n_sensors}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Generate sensor files with "
                "`python -m src.sensors.generate_sensors` from the repository root."
            )
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        if not is_current_podqr_record(rec, n_sensors):
            raise ValueError(
                f"{path} is stale (does not match sensors/layouts/podqr_layout.json). "
                "Regenerate it with `python -m src.sensors.generate_sensors`."
            )
        pos = np.asarray(rec["positions"], dtype=np.float64)
        if positions is None:
            positions = pos
            noise = SensorNoise(
                sigma_C=float(rec["noise_sigma_C"]),
                quant_step_C=float(rec["quant_step_C"]),
                offsets_C=np.asarray(rec["calibration_offsets_C"], dtype=np.float64),
            )
        elif not np.allclose(pos, positions):
            raise ValueError(f"{path}: sensor positions differ between chips")
        true_rows.append(rec["temperatures_true"])
        noisy_rows.append(rec["temperatures_noisy"])

    T = Q = None
    if with_fields:
        mat = raw_dir / split / "matrices"
        T = np.stack([np.load(str(mat / f"{sid}.npy")) for sid in ids]).astype(np.float32)
        Q = np.stack([np.load(str(mat / f"{sid}_Q.npy")) for sid in ids]).astype(np.float32)
        if T.shape[1:] != (GRID, GRID) or Q.shape != T.shape:
            raise ValueError(f"unexpected field shape {T.shape} / {Q.shape}")

    return SplitArrays(
        split=split, n_sensors=n_sensors, ids=list(ids), T_amb=T_amb, positions_m=positions,
        sensors_true=np.asarray(true_rows, dtype=np.float64),
        sensors_noisy=np.asarray(noisy_rows, dtype=np.float64),
        noise=noise, T=T, Q=Q,
    )
