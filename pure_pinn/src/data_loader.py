"""Loading and normalisation for the HotSpot-style steady-state thermal dataset.

Dataset facts this module encodes (see ``PROTOCOL.md`` and ``generator/pde_solver.py``):

* Governing equation ``-div(t_die k(T) grad T) + h_eff (T - T_amb) = q(x, y, T)``
  on a 12 mm x 12 mm die, with adiabatic (Neumann) lateral edges and a
  distributed vertical heat sink ``h_eff``. ``k`` depends on the local
  temperature, so it is *not* stored in the metadata.
* ``T_amb`` is per-chip (``T_amb_C`` in the metadata) and is a known constant of
  the model, not something to be inferred.
* The heat sources are axis-aligned blocks (``blocks_json``) with a uniform
  dynamic power density and a temperature-dependent leakage term. The stored
  ``thermal_NNNNN_Q.npy`` is the total source ``q`` in W/m^2 at the converged
  temperature.
* Matrices are ``128 x 128`` with ``M[i, j] = T(x = j/(n-1), y = i/(n-1))``, so
  the row index is *y* and the column index is *x*. Coordinates below are
  normalised to the unit square (``x_hat = x / L``).
* Sensor files live in ``data/generated/<split>/`` and hold POD-QR placements;
  the model input is ``temperatures_noisy`` only (``PROTOCOL.md``).
* Temperatures are in degC (peak 30-125 degC across the 700 chips).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils import resolve_project_path

SPLITS = ("train", "validation", "test")
SPLIT_ALIASES = {"val": "validation"}
SENSOR_COUNTS = (5, 8, 10)
DIE_L = 0.012  # m


def normalise_split(split: str) -> str:
    """Accept ``val`` as an alias of ``validation``."""
    split = SPLIT_ALIASES.get(split, split)
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {SPLITS}")
    return split


# --------------------------------------------------------------------------- #
# per-chip record
# --------------------------------------------------------------------------- #
@dataclass
class ChipSample:
    """Everything known about one chip, with ground truth kept separate."""

    sample_id: str
    split: str
    T_amb: float                 # known ambient temperature, degC
    die_L: float                 # die side length, m
    peak_temperature: float      # ground truth, evaluation only
    min_temperature: float       # ground truth, evaluation only
    block_count: int             # ground truth, evaluation only
    blocks: list                 # ground-truth block dicts, evaluation only
    root: Path                   # data/raw/dataset
    sensor_root: Path            # data/generated

    # ---------------- inputs a physics-only model is allowed to see -------- #
    def sensors(self, n_sensors: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions (N, 2) in [0,1]^2, noisy temperatures (N,) degC)``."""
        path = self.sensor_root / self.split / f"{self.sample_id}_sensors_{n_sensors}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Generate sensor files with "
                "`python -m src.sensors.generate_sensors` from the repository root."
            )
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        pos = np.asarray(payload["positions"], dtype=np.float64) / self.die_L
        temp = np.asarray(payload["temperatures_noisy"], dtype=np.float64)
        if pos.shape != (n_sensors, 2) or temp.shape != (n_sensors,):
            raise ValueError(f"{path} does not hold {n_sensors} sensors")
        return pos, temp

    # ---------------- ground truth, evaluation only ------------------------ #
    def field(self) -> np.ndarray:
        """Full ``(128, 128)`` temperature field. Never a training signal."""
        path = self.root / self.split / "matrices" / f"{self.sample_id}.npy"
        return np.load(path).astype(np.float64)

    def source_field(self) -> np.ndarray:
        """Ground-truth total source ``q(x, y)`` on the 128 grid, W/m^2."""
        path = self.root / self.split / "matrices" / f"{self.sample_id}_Q.npy"
        return np.load(path).astype(np.float64)

    def block_centres(self) -> np.ndarray:
        """``(M, 2)`` block centres in ``[0,1]^2``, for validating an inferred source."""
        if not self.blocks:
            return np.zeros((0, 2))
        return np.array([[(b["x_min"] + b["x_max"]) / 2 / self.die_L,
                          (b["y_min"] + b["y_max"]) / 2 / self.die_L] for b in self.blocks])

    def block_powers(self) -> np.ndarray:
        """``(M,)`` total power of each block at 25 degC reference, W."""
        return np.array([(b["power_density_dynamic"] + b["power_density_leakage0"]) * 1e4
                         * b["width"] * b["height"] for b in self.blocks])


# --------------------------------------------------------------------------- #
# metadata / index
# --------------------------------------------------------------------------- #
def default_data_root() -> Path:
    """The dataset committed under ``data/raw/dataset``."""
    return resolve_project_path("data/raw/dataset")


def default_sensor_root() -> Path:
    """POD-QR sensor records committed under ``data/generated``."""
    return resolve_project_path("data/generated")


@lru_cache(maxsize=None)
def load_metadata(root: str, split: str) -> pd.DataFrame:
    """Index file for one split, with ``blocks_json`` already parsed."""
    split = normalise_split(split)
    df = pd.read_csv(Path(root) / split / "metadata.csv")
    df["blocks"] = df["blocks_json"].map(json.loads)
    return df.set_index("sample_id", drop=False)


def chip_ids(root: str | Path | None, split: str) -> list[str]:
    root = str(default_data_root() if root is None else Path(root))
    return list(load_metadata(root, split).index)


def load_chip(sample_id: str, split: str, root: str | Path | None = None,
              sensor_root: str | Path | None = None) -> ChipSample:
    split = normalise_split(split)
    root = default_data_root() if root is None else Path(root)
    sensor_root = default_sensor_root() if sensor_root is None else Path(sensor_root)
    df = load_metadata(str(root), split)
    if sample_id not in df.index:
        raise KeyError(f"{sample_id} is not in the {split} split ({len(df)} chips)")
    row = df.loc[sample_id]
    return ChipSample(
        sample_id=sample_id,
        split=split,
        T_amb=float(row["T_amb_C"]),
        die_L=float(row["die_L_x_m"]),
        peak_temperature=float(row["peak_temp_C"]),
        min_temperature=float(row["min_temp_C"]),
        block_count=int(row["block_count"]),
        blocks=list(row["blocks"]),
        root=root,
        sensor_root=sensor_root,
    )


# --------------------------------------------------------------------------- #
# grid helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def grid_coords(n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """``(X, Y)`` with ``X[i, j] = j/(n-1)`` and ``Y[i, j] = i/(n-1)``."""
    axis = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(axis, axis)
    X.setflags(write=False)
    Y.setflags(write=False)
    return X, Y


def grid_points(n: int = 128) -> np.ndarray:
    """``(n*n, 2)`` array of ``(x, y)`` in the same row-major order as the matrix."""
    X, Y = grid_coords(n)
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def field_from_points(values: np.ndarray, n: int = 128) -> np.ndarray:
    """Inverse of :func:`grid_points`: fold a flat prediction back to ``(n, n)``."""
    return np.asarray(values, dtype=np.float64).reshape(n, n)


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
@dataclass
class ChipScaler:
    """Per-chip affine scaling ``theta = (T - T_amb) / scale``.

    The offset is the chip's known ambient temperature (the sink temperature in
    the governing equation); the scale is the hottest sensor's rise above
    ambient. Both come from metadata and the sparse readings, never from the
    ground-truth field, so no evaluation signal leaks into training.
    """

    offset: float
    scale: float

    @classmethod
    def from_sensors(cls, chip: ChipSample, sensor_temps: np.ndarray, floor: float = 1.0) -> "ChipScaler":
        span = float(np.max(sensor_temps) - chip.T_amb)
        return cls(offset=chip.T_amb, scale=max(span, floor))

    def normalise(self, T: np.ndarray) -> np.ndarray:
        return (np.asarray(T, dtype=np.float64) - self.offset) / self.scale

    def denormalise(self, theta):
        return theta * self.scale + self.offset


@dataclass
class GlobalScaler:
    """Dataset-level min-max scaling, for cross-chip models (A and C).

    Fitted from the train-split index only, so it needs no matrix reads.
    """

    t_min: float
    t_max: float

    @classmethod
    def fit(cls, root: str | Path | None = None, split: str = "train") -> "GlobalScaler":
        root = default_data_root() if root is None else Path(root)
        df = load_metadata(str(root), split)
        return cls(t_min=float(df["min_temp_C"].min()), t_max=float(df["peak_temp_C"].max()))

    def normalise(self, T):
        return (T - self.t_min) / (self.t_max - self.t_min)

    def denormalise(self, T_norm):
        return T_norm * (self.t_max - self.t_min) + self.t_min
