"""POD-QR sensor placement for arbitrary-size thermal maps (Manohar et al. 2018), fitted on training maps only."""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def pod_basis(train_theta: np.ndarray, energy: float = 0.95, min_modes: int = 10):
    """``train_theta``: ``(n, H, W)`` normalised maps. Returns ``(mean (H*W,), basis (H*W, r))``.

    Uses the n x n Gram matrix (cheap when n << H*W). ``r`` keeps ``energy`` of the variance but never fewer than ``min_modes``
    (QR pivoting only has ``r`` informative pivots, so ``r`` must be at least the largest sensor count).
    """
    n = train_theta.shape[0]
    X = train_theta.reshape(n, -1).astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    lam, V = np.linalg.eigh(Xc @ Xc.T)
    lam, V = np.clip(lam[::-1], 0, None), V[:, ::-1]
    r = max(int(np.searchsorted(np.cumsum(lam) / lam.sum(), energy)) + 1, min_modes)
    r = min(r, int((lam > 1e-9 * lam[0]).sum()))
    basis = Xc.T @ V[:, :r] / np.sqrt(np.maximum(lam[:r], 1e-12))
    return mean, basis


def podqr_sensors(basis: np.ndarray, shape: tuple, n_max: int = 10, margin: int = 3) -> np.ndarray:
    """First ``n_max`` QR-pivot pixels of the POD basis, away from the edges. Returns ``(n_max, 2)`` of ``(row, col)``.

    Nested by construction: the first 5 are the 5-sensor layout, the first 8 the 8-sensor layout.
    """
    H, W = shape
    allowed = np.zeros((H, W), dtype=bool)
    allowed[margin:H - margin, margin:W - margin] = True
    idx = np.flatnonzero(allowed.ravel())
    _, _, piv = qr(basis[idx].T, pivoting=True, mode="economic")
    flat = idx[piv[:n_max]]
    return np.stack(np.unravel_index(flat, (H, W)), axis=1)
