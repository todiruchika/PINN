"""Physics-based reconstruction of a thermal map from a few sensors (PINN-style inverse problem, no neural network).

Model: steady heat conduction in the die with vertical heat loss and adiabatic edges,

        -kappa * laplacian(T) + h * (T - T0) = q(x, y),        dT/dn = 0 on the edges,

i.e. ``(1 - ell^2 * laplacian) (T - T0) = q / h`` with ``ell^2 = kappa / h`` (a length in pixels). The heat sources ``q`` are
unknown. We write them as a sum of Gaussian bumps on a coarse grid, solve the equation for each bump with a cosine transform
(which gives the adiabatic edges), and fit the bump amplitudes to the noisy sensor readings with a ridge penalty. T0 is a
free constant. The map is then ``T = T0 + Phi @ a``. No training maps are used; only ``ell``, the bump size and the penalty are
tuned (on validation frames), like the physical constants of a PINN.

The same machinery can also *correct* any other prediction: fit the physics response to the sensor residuals
(``y - prediction at the sensors``), which spreads the mismatch over the map in a way the heat equation allows.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn


def build_response(shape: tuple, ell: float, spacing: int) -> np.ndarray:
    """``Phi (H*W, B)``: temperature response of the heat equation to a unit Gaussian source bump at each grid centre."""
    H, W = shape
    ys = np.arange(spacing // 2, H, spacing)
    xs = np.arange(spacing // 2, W, spacing)
    sig = spacing / 1.2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    ky = (np.pi * np.arange(H) / H) ** 2
    kx = (np.pi * np.arange(W) / W) ** 2
    mult = 1.0 / (1.0 + ell ** 2 * (ky[:, None] + kx[None, :]))
    cols = []
    for cy in ys:
        for cx in xs:
            bump = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
            cols.append(idctn(mult * dctn(bump, type=2, norm="ortho"), type=2, norm="ortho").ravel())
    return np.stack(cols, axis=1).astype(np.float32)


def reconstruct(Phi: np.ndarray, flat_idx: np.ndarray, y: np.ndarray, lam: float, base: np.ndarray | None = None) -> np.ndarray:
    """Fit ``y`` (readings at ``flat_idx``) with ``T0 + Phi a`` (ridge on ``a``, free constant ``T0``).

    If ``base`` (a full map, flattened) is given, the physics only models the residual ``y - base[flat_idx]``.
    Returns the full flattened map (``base + T0 + Phi a`` or ``T0 + Phi a``).
    """
    R = Phi[flat_idx].astype(np.float64)                  # (n, B)
    target = y - (base[flat_idx] if base is not None else 0.0)
    n = len(y)
    K = R @ R.T
    scale = np.trace(K) / n
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K + lam * scale * np.eye(n)
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    sol = np.linalg.solve(A, np.concatenate([target, [0.0]]))
    alpha, c = sol[:n], sol[n]
    out = c + Phi @ (R.T @ alpha).astype(np.float32)
    return out + base if base is not None else out
