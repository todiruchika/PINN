"""Sensor layout functions returning pixel (row, col) index lists."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

LAYOUT_DIR = Path("sensors/layouts")
DIE_L = 0.012
GRID = 128


def podqr_layout(n_sensors: int) -> list:
    """Return first n_sensors POD-QR pivot positions as (row, col)."""
    with open(LAYOUT_DIR / "podqr_layout.json") as f:
        layout = json.load(f)
    return [(p["row"], p["col"]) for p in layout["pivots"][:n_sensors]]


def random_uniform_layout(n_sensors: int, seed: int, margin: int = 3) -> list:
    """Uniform random layout with margin, seeded."""
    rng = np.random.default_rng(seed)
    lo, hi = margin, GRID - margin
    rows = rng.integers(lo, hi, size=n_sensors)
    cols = rng.integers(lo, hi, size=n_sensors)
    return list(zip(rows.tolist(), cols.tolist()))


def block_centre_legacy_layout(sample_id: str, n_sensors: int, raw_dir: Path = Path("data/raw/dataset")) -> list:
    """Return legacy block_centre_with_jitter sensor positions from raw JSON."""
    from src.sensors.readers import metre_to_idx
    for split in ["train", "validation", "test"]:
        p = raw_dir / split / "sensors" / f"{sample_id}_sensors_{n_sensors}.json"
        if p.exists():
            with open(p) as f:
                s = json.load(f)
            return [
                (metre_to_idx(pos[1]), metre_to_idx(pos[0]))
                for pos in s["positions"]
            ]
    raise FileNotFoundError(f"No legacy sensor file for {sample_id} N={n_sensors}")


def kmeans_floorplan_layout(n_sensors: int, floorplan_metadata: dict) -> list:
    """STUB: K-means seeded from floorplan centroids. Layout only, never a model input.

    This is a placeholder for the k-means floorplan sensor placement strategy.
    It uses floorplan metadata (block centroids) to seed initial cluster centres,
    then places sensors at k-means cluster centres over the grid.

    Not yet implemented — pending floorplan metadata parsing convention.
    """
    raise NotImplementedError("kmeans_floorplan is a stub — not yet implemented")
