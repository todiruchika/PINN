"""Linear first estimate of the temperature field from the sensor readings.

The estimate is a POD (proper orthogonal decomposition) fit: the chip's sensor readings are
matched to a set of basis maps, and the best-fitting combination is returned as the full map.

Leakage rules:
- the basis and mean come from the TRAINING chips only;
- at test time the estimate uses only the sensor readings and the ambient temperature.
"""
from __future__ import annotations

import numpy as np
import torch

from src.sensors.readers import metre_to_idx

GRID = 128
TEMP_SCALE_C = 50.0


def fit_basis(train_T: np.ndarray, train_ambient: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and top-`rank` POD modes of the training rises above ambient.

    train_T: (n, 128, 128) degC; train_ambient: (n,) degC.
    Returns mean (16384,) and basis (16384, rank), both in degC of rise.
    """
    X = (train_T - train_ambient.reshape(-1, 1, 1)).reshape(len(train_T), -1).T   # (16384, n)
    mean = X.mean(axis=1, keepdims=True)
    U, _, _ = np.linalg.svd(X - mean, full_matrices=False)
    return mean[:, 0], U[:, :rank]


def prior_theta(
    positions_m: np.ndarray,
    readings_c: np.ndarray,
    ambient_c: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> torch.Tensor:
    """Dimensionless first estimate theta = (T_est - T_amb) / S, shape (n, 128, 128).

    positions_m: (n, N, 2) sensor (x, y) in metres; readings_c: (n, N) noisy degC;
    ambient_c: (n,) degC. Uses only quantities a test chip provides.
    """
    out = np.zeros((len(readings_c), GRID * GRID), dtype=np.float32)
    for i in range(len(readings_c)):
        idx = np.array([metre_to_idx(y) * GRID + metre_to_idx(x) for x, y in positions_m[i]])
        rise = readings_c[i] - ambient_c[i]
        coeffs = np.linalg.lstsq(basis[idx], rise - mean[idx], rcond=1e-3)[0]
        out[i] = mean + basis @ coeffs
    return torch.tensor(out.reshape(-1, GRID, GRID) / TEMP_SCALE_C, dtype=torch.float32)
