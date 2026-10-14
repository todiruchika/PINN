"""Load a sample by ID from data/raw/dataset/."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

RAW_DIR = Path("data/raw/dataset")
GENERATED_SENSOR_DIR = Path("data/generated")
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
    generated_dir: Path = GENERATED_SENSOR_DIR,
) -> dict:
    """Return T grid (128x128), Q grid, metadata row, and sensor arrays."""
    if meta is None:
        meta = load_metadata(raw_dir)

    split = _split_of(sample_id, meta)
    split_dir = raw_dir / split

    T = np.load(str(split_dir / "matrices" / f"{sample_id}.npy"))
    Q = np.load(str(split_dir / "matrices" / f"{sample_id}_Q.npy"))

    if strategy == "block_centre_with_jitter":
        strategy = "block_centre_reference"

    if strategy == "podqr":
        sensor_path = generated_dir / split / f"{sample_id}_sensors_{n_sensors}.json"
        if not sensor_path.exists():
            raise FileNotFoundError(
                f"POD-QR sensors are missing: {sensor_path}. "
                "Generate them with `python -m src.sensors.generate_sensors`."
            )
        with open(sensor_path) as f:
            sensors = json.load(f)
        from src.sensors.layouts import is_current_podqr_record
        if not is_current_podqr_record(sensors, n_sensors):
            raise ValueError(
                f"POD-QR sensor data is stale or does not match the current layout: {sensor_path}. "
                "Regenerate it with `python -m src.sensors.generate_sensors`."
            )
    elif strategy == "block_centre_reference":
        sensor_path = split_dir / "sensors" / f"{sample_id}_sensors_{n_sensors}.json"
        if not sensor_path.exists():
            raise FileNotFoundError(f"Reference sensor file is missing: {sensor_path}")
        with open(sensor_path) as f:
            sensors = json.load(f)
    elif strategy == "random_uniform":
        from src.sensors.generate_sensors import sensor_record_for_layout
        from src.sensors.layouts import random_uniform_layout

        sample_num = int(sample_id.rsplit("_", 1)[1])
        layout_seed = 42 + sample_num * 31 + n_sensors
        layout = random_uniform_layout(n_sensors, seed=layout_seed)
        sensors = sensor_record_for_layout(
            T, layout, n_sensors, "random_uniform", sample_num
        )
    elif strategy == "kmeans_floorplan":
        raise NotImplementedError("kmeans_floorplan sensor placement is not implemented")
    else:
        raise ValueError(
            f"Unknown sensor strategy {strategy!r}; expected podqr, "
            "random_uniform, block_centre_reference, or kmeans_floorplan"
        )

    meta_row = meta.loc[meta["sample_id"] == sample_id].iloc[0].to_dict()
    return {"T": T, "Q": Q, "meta": meta_row, "sensors": sensors}
