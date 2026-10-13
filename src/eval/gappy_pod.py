"""Gappy POD reconstruction — non-neural baseline.

Given the saved POD basis (from podqr_basis.npz) and sensor readings at pivot pixels,
reconstruct the full 128x128 temperature field.
"""
import numpy as np
from pathlib import Path

BASIS_PATH = Path("sensors/layouts/podqr_basis.npz")


def load_basis(basis_path: Path = BASIS_PATH):
    data = np.load(str(basis_path))
    return data["mean"], data["basis"]


def reconstruct(
    sensor_values: np.ndarray,
    sensor_flat_indices: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Reconstruct full field from sensor readings.

    Args:
        sensor_values: (n_sensors,) temperature values at sensor locations
        sensor_flat_indices: (n_sensors,) flat indices into 16384-vector
        mean: (16384,) training mean
        basis: (16384, r) POD basis

    Returns:
        (128, 128) reconstructed temperature field
    """
    Psi_s = basis[sensor_flat_indices, :]  # (n_sensors, r)
    y = sensor_values - mean[sensor_flat_indices]  # (n_sensors,)
    coeffs, _, _, _ = np.linalg.lstsq(Psi_s, y, rcond=None)
    field = mean + basis @ coeffs
    return field.reshape(128, 128)


def gappy_pod_predict(
    sample_id: str,
    n_sensors: int,
    raw_dir: Path = Path("data/raw/dataset"),
    basis_path: Path = BASIS_PATH,
    use_generated: bool = True,
) -> tuple:
    """Load basis, sensor readings, reconstruct, return (pred, true)."""
    import json
    from src.sensors.readers import metre_to_idx

    mean, basis = load_basis(basis_path)

    import pandas as pd
    meta = pd.read_csv(raw_dir / "metadata.csv")
    row = meta.loc[meta["sample_id"] == sample_id].iloc[0]
    split = row["split"]

    T_true = np.load(str(raw_dir / split / "matrices" / f"{sample_id}.npy"))

    gen_path = Path("data/generated") / split / f"{sample_id}_sensors_{n_sensors}.json"
    raw_path = raw_dir / split / "sensors" / f"{sample_id}_sensors_{n_sensors}.json"
    sensor_path = gen_path if (use_generated and gen_path.exists()) else raw_path

    with open(sensor_path) as f:
        sensors = json.load(f)

    flat_indices = np.array([
        metre_to_idx(pos[1]) * 128 + metre_to_idx(pos[0])
        for pos in sensors["positions"]
    ])
    sensor_values = np.array(sensors["temperatures_noisy"])

    pred = reconstruct(sensor_values, flat_indices, mean, basis)
    return pred, T_true
