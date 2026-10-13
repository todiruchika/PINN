"""Load a sample by ID from data/raw/dataset/."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

RAW_DIR = Path("data/raw/dataset")
DIE_L = 0.012
GRID = 128


def _split_of(sample_id: str, meta: pd.DataFrame) -> str:
    row = meta.loc[meta["sample_id"] == sample_id]
    if row.empty:
        raise KeyError(f"Unknown sample_id: {sample_id}")
    return row.iloc[0]["split"]


def load_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "metadata.csv")


def load_sample(
    sample_id: str,
    n_sensors: int,
    strategy: str = "podqr",
    raw_dir: Path = RAW_DIR,
    meta: Optional[pd.DataFrame] = None,
) -> dict:
    """Return T grid (128x128), Q grid, metadata row, and sensor arrays."""
    if meta is None:
        meta = load_metadata(raw_dir)

    split = _split_of(sample_id, meta)
    split_dir = raw_dir / split

    T = np.load(str(split_dir / "matrices" / f"{sample_id}.npy"))
    Q = np.load(str(split_dir / "matrices" / f"{sample_id}_Q.npy"))

    sensor_dir = split_dir / "sensors"
    gen_dir = Path("data/generated") / split
    if strategy == "podqr" and (gen_dir / f"{sample_id}_sensors_{n_sensors}.json").exists():
        sensor_path = gen_dir / f"{sample_id}_sensors_{n_sensors}.json"
    else:
        sensor_path = sensor_dir / f"{sample_id}_sensors_{n_sensors}.json"

    with open(sensor_path) as f:
        sensors = json.load(f)

    meta_row = meta.loc[meta["sample_id"] == sample_id].iloc[0].to_dict()
    return {"T": T, "Q": Q, "meta": meta_row, "sensors": sensors}
