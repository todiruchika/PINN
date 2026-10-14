"""Finite-difference PDE residual of a 128x128 temperature field, in physical units.

    R = -t_die * div( k(T) grad T ) + h_eff * (T - T_amb) - Q            [W/m^2]

with the dataset's ground-truth source ``Q`` (``thermal_NNNNN_Q.npy``, total
source incl. leakage). ``relative_pde_residual`` is ``||R|| / ||Q||`` over the
interior nodes (one cell in from the edge), so it measures how far a *predicted*
map is from satisfying the governing equation. It is a scoring metric only; no
model uses it for training. The residual of the ground-truth map is the
discretisation floor (the data come from a 512 mesh downsampled to 128).
"""
from __future__ import annotations

import numpy as np

from src.physics.losses import Physics


def _k_np(T_C: np.ndarray, p: Physics) -> np.ndarray:
    T_K = np.clip(T_C + 273.15, 250.0, 1200.0)
    return np.clip(p.k_A / (T_K - p.k_offset) + p.k_B, p.k_min, p.k_max)


def pde_residual_field(T_C: np.ndarray, Q: np.ndarray, T_amb: float, physics: Physics | None = None) -> np.ndarray:
    """Residual ``R`` (W/m^2) on interior nodes, shape ``(n-2, n-2)``."""
    p = physics or Physics()
    n = T_C.shape[0]
    d = p.L / (n - 1)
    k = _k_np(T_C, p)
    kx = 0.5 * (k[:, 1:] + k[:, :-1])            # faces between columns, (n, n-1)
    ky = 0.5 * (k[1:, :] + k[:-1, :])            # faces between rows,    (n-1, n)
    fx = kx * (T_C[:, 1:] - T_C[:, :-1]) / d
    fy = ky * (T_C[1:, :] - T_C[:-1, :]) / d
    div = (fx[1:-1, 1:] - fx[1:-1, :-1]) / d + (fy[1:, 1:-1] - fy[:-1, 1:-1]) / d
    return -p.t_die * div + p.h_eff * (T_C[1:-1, 1:-1] - T_amb) - Q[1:-1, 1:-1]


def relative_pde_residual(T_C: np.ndarray, Q: np.ndarray, T_amb: float, physics: Physics | None = None) -> float:
    R = pde_residual_field(T_C, Q, T_amb, physics)
    return float(np.linalg.norm(R) / (np.linalg.norm(Q[1:-1, 1:-1]) + 1e-12))
