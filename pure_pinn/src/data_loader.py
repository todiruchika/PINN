"""Loading and normalisation for the semiconductor steady-state thermal dataset.

Dataset facts this module encodes (verified against the generated data, not
just the README):

* Governing equation is ``-k \\nabla^2 T = Q`` on ``[0,1]^2``, i.e.
  ``\\nabla^2 T = -Q/k``.
* ``k`` is per-chip in ``[100, 150]`` -- it is *not* shared across chips.
* Dirichlet boundary ``T = T_boundary`` with ``T_boundary`` per-chip in
  ``[20, 40] degC``; it equals ``min_temperature`` to ~5e-5 and the matrix
  border reproduces it exactly.
* ``hotspot_positions`` are axis-aligned rectangles ``[x0, y0, x1, y1]``, each
  with a uniform power density from ``hotspot_power_values`` (W/m^2).
* Matrices are ``128 x 128`` with ``M[i, j] = T(x = j/(n-1), y = i/(n-1))``,
  so row index is *y* and column index is *x*.
* Sensor positions land exactly on grid nodes.
"""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from utils import resolve_path

SPLITS = ("train", "val", "test")
SENSOR_COUNTS = (5, 8, 10)


# --------------------------------------------------------------------------- #
# per-chip record
# --------------------------------------------------------------------------- #
@dataclass
class ChipSample:
    """Everything known about one chip, with ground truth kept separate."""

    sample_id: str
    split: str
    k: float                     # thermal conductivity, W/(m K)
    T_boundary: float            # Dirichlet value on the whole of dOmega, degC
    peak_temperature: float      # ground truth, evaluation only
    min_temperature: float       # ground truth, evaluation only
    hotspot_count: int           # ground truth, evaluation only
    hotspots: np.ndarray         # (M, 4) [x0, y0, x1, y1], ground truth
    powers: np.ndarray           # (M,) W/m^2, ground truth
    root: Path

    # ---------------- inputs a physics-only model is allowed to see -------- #
    def sensors(self, n_sensors: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(positions (N, 2) in [0,1]^2, temperatures (N,) degC)``."""
        path = self.root / self.split / "sensors" / f"{self.sample_id}_sensors_{n_sensors}.json"
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        pos = np.asarray(payload["positions"], dtype=np.float64)
        temp = np.asarray(payload["temperatures"], dtype=np.float64)
        if pos.shape != (n_sensors, 2) or temp.shape != (n_sensors,):
            raise ValueError(f"{path} does not hold {n_sensors} sensors")
        return pos, temp

    # ---------------- ground truth, evaluation only ------------------------ #
    def field(self) -> np.ndarray:
        """Full ``(128, 128)`` temperature field. Never a training signal for Model B."""
        path = self.root / self.split / "matrices" / f"{self.sample_id}.npy"
        return np.load(path).astype(np.float64)

    def source_field(self, n: int = 128) -> np.ndarray:
        """Ground-truth ``Q(x, y)`` rasterised on the ``n x n`` grid, W/m^2."""
        X, Y = grid_coords(n)
        Q = np.zeros((n, n), dtype=np.float64)
        for (x0, y0, x1, y1), q in zip(self.hotspots, self.powers):
            Q += q * ((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
        return Q

    def hotspot_centres(self) -> np.ndarray:
        """``(M, 2)`` rectangle centres, for validating an inferred source field."""
        if len(self.hotspots) == 0:
            return np.zeros((0, 2))
        return np.stack(
            [(self.hotspots[:, 0] + self.hotspots[:, 2]) / 2,
             (self.hotspots[:, 1] + self.hotspots[:, 3]) / 2],
            axis=1,
        )


# --------------------------------------------------------------------------- #
# metadata / index
# --------------------------------------------------------------------------- #
def default_data_root() -> Path:
    """The dataset shipped alongside this repo."""
    return resolve_path("../dataset/py_pipeline/dataset")


@lru_cache(maxsize=None)
def load_metadata(root: str, split: str) -> pd.DataFrame:
    """Index file for one split, with list columns already parsed."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    df = pd.read_csv(Path(root) / split / "metadata.csv")
    df["hotspot_positions"] = df["hotspot_positions"].map(ast.literal_eval)
    df["hotspot_power_values"] = df["hotspot_power_values"].map(ast.literal_eval)
    return df.set_index("sample_id", drop=False)


def chip_ids(root: str | Path | None, split: str) -> list[str]:
    root = str(default_data_root() if root is None else Path(root))
    return list(load_metadata(root, split).index)


def load_chip(sample_id: str, split: str, root: str | Path | None = None) -> ChipSample:
    root = default_data_root() if root is None else Path(root)
    row = load_metadata(str(root), split).loc[sample_id]
    return ChipSample(
        sample_id=sample_id,
        split=split,
        k=float(row["thermal_conductivity"]),
        T_boundary=float(row["boundary_temperature"]),
        peak_temperature=float(row["peak_temperature"]),
        min_temperature=float(row["min_temperature"]),
        hotspot_count=int(row["hotspot_count"]),
        hotspots=np.asarray(row["hotspot_positions"], dtype=np.float64).reshape(-1, 4),
        powers=np.asarray(row["hotspot_power_values"], dtype=np.float64).reshape(-1),
        root=Path(root),
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
    """Per-chip affine scaling ``T_norm = (T - offset) / scale``.

    Built from information a physics-only model legitimately has: the chip's
    boundary temperature (metadata) and its sparse sensor readings. It never
    touches the ground-truth field, so no evaluation signal leaks into
    training.

    The offset is ``T_boundary``, which makes the hard-BC construction
    homogeneous in normalised units (``T_norm = 0`` on ``dOmega``).
    """

    offset: float
    scale: float

    @classmethod
    def from_sensors(cls, chip: ChipSample, sensor_temps: np.ndarray, floor: float = 1.0) -> "ChipScaler":
        span = float(np.max(sensor_temps) - chip.T_boundary)
        return cls(offset=chip.T_boundary, scale=max(span, floor))

    def normalise(self, T: np.ndarray) -> np.ndarray:
        return (np.asarray(T, dtype=np.float64) - self.offset) / self.scale

    def denormalise(self, T_norm):
        return T_norm * self.scale + self.offset

    def source_scale(self, k: float) -> float:
        """Factor turning physical ``Q`` into the normalised forcing ``f_norm``.

        ``\\nabla^2 T = -Q/k`` becomes ``\\nabla^2 T_norm = -f_norm`` with
        ``f_norm = Q / (k * scale)``, so ``Q = f_norm * k * scale``.
        """
        return 1.0 / (k * self.scale)


@dataclass
class GlobalScaler:
    """Dataset-level min-max scaling, for the cross-chip models (A and C).

    Fitted from the train-split index only (``min_temperature`` /
    ``peak_temperature`` columns), so it needs no matrix reads.
    """

    t_min: float
    t_max: float

    @classmethod
    def fit(cls, root: str | Path | None = None, split: str = "train") -> "GlobalScaler":
        root = default_data_root() if root is None else Path(root)
        df = load_metadata(str(root), split)
        return cls(t_min=float(df["min_temperature"].min()),
                   t_max=float(df["peak_temperature"].max()))

    def normalise(self, T):
        return (T - self.t_min) / (self.t_max - self.t_min)

    def denormalise(self, T_norm):
        return T_norm * (self.t_max - self.t_min) + self.t_min


# --------------------------------------------------------------------------- #
# split bookkeeping
# --------------------------------------------------------------------------- #
def write_split_lists(root: str | Path | None = None, out_dir: str | Path = "data/splits") -> dict[str, int]:
    """Persist the chip-ID lists for each split so runs are reproducible."""
    root = default_data_root() if root is None else Path(root)
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        ids = chip_ids(root, split)
        (out / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        counts[split] = len(ids)
    return counts


def read_split_list(split: str, out_dir: str | Path = "data/splits") -> list[str]:
    path = resolve_path(out_dir) / f"{split}.txt"
    return path.read_text(encoding="utf-8").split()
