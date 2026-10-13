"""POD-QR sensor placement (Manohar et al. 2018).

Reads ONLY the train split. Saves:
  sensors/layouts/podqr_layout.json
  sensors/layouts/podqr_basis.npz
  figures/podqr_modes.png
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from pathlib import Path

from src.data.dataset import RAW_DIR, GRID

LAYOUT_DIR = Path("sensors/layouts")
FIGURES_DIR = Path("figures")
DIE_L = 0.012
DEFAULT_MARGIN = 3
DEFAULT_ENERGY = 0.95


def _T_b(T: np.ndarray) -> float:
    """Mean of four edge temperatures (default T_b rule; PENDING until BC known)."""
    edge = np.concatenate([T[0, :], T[-1, :], T[1:-1, 0], T[1:-1, -1]])
    return float(edge.mean())


def _qr_pivot(A: np.ndarray):
    """QR with column pivoting via scipy."""
    from scipy.linalg import qr
    Q, R, P = qr(A.T, pivoting=True)
    return Q, R, P


def fit_podqr_layout(
    train_ids: list,
    raw_dir: Path = RAW_DIR,
    energy_threshold: float = DEFAULT_ENERGY,
    n_sensors: int = 10,
    margin: int = DEFAULT_MARGIN,
    seed: int = 42,
) -> dict:
    """Fit POD-QR layout from training samples only. Returns layout dict."""
    split = "train"
    N = GRID

    snapshots = []
    for sid in sorted(train_ids):
        T = np.load(str(raw_dir / split / "matrices" / f"{sid}.npy")).astype(np.float64)
        Tb = _T_b(T)
        Tmax = T.max()
        denom = Tmax - Tb if (Tmax - Tb) > 1e-10 else 1.0
        theta = (T - Tb) / denom
        snapshots.append(theta.ravel())

    X = np.stack(snapshots, axis=1)  # (16384, n_train)
    mean = X.mean(axis=1, keepdims=True)
    X_c = X - mean

    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    cumulative = np.cumsum(s ** 2) / (s ** 2).sum()
    r = int(np.searchsorted(cumulative, energy_threshold)) + 1
    Psi = U[:, :r]  # (16384, r)

    mask = np.zeros((N, N), dtype=bool)
    mask[margin:N - margin, margin:N - margin] = True
    valid_flat = np.where(mask.ravel())[0]

    Psi_valid = Psi[valid_flat, :]  # (n_valid, r)
    _, _, pivot_in_valid = _qr_pivot(Psi_valid)
    sensor_flat_in_valid = pivot_in_valid[:n_sensors]
    sensor_flat = valid_flat[sensor_flat_in_valid]

    rows, cols = np.unravel_index(sensor_flat, (N, N))
    positions_m = [
        [float(cols[i] * DIE_L / (N - 1)), float(rows[i] * DIE_L / (N - 1))]
        for i in range(n_sensors)
    ]

    sha = hashlib.sha256(",".join(sorted(train_ids)).encode()).hexdigest()

    layout = {
        "pivots": [{"row": int(rows[i]), "col": int(cols[i])} for i in range(n_sensors)],
        "positions_m": positions_m,
        "r": r,
        "energy_threshold": energy_threshold,
        "cumulative_energy_at_r": float(cumulative[r - 1]),
        "T_b_rule": "mean_edge",
        "boundary_margin_cells": margin,
        "train_ids_sha256": sha,
    }

    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAYOUT_DIR / "podqr_layout.json", "w") as f:
        json.dump(layout, f, indent=2)

    np.savez(str(LAYOUT_DIR / "podqr_basis.npz"), mean=mean.ravel(), basis=Psi)
    print(f"Saved layout: r={r}, energy={cumulative[r-1]:.4f}, margin={margin}")

    _save_mode_figure(Psi, mean, N)
    return layout


def _save_mode_figure(Psi: np.ndarray, mean: np.ndarray, N: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, ax in enumerate(axes):
            if i < Psi.shape[1]:
                mode = Psi[:, i].reshape(N, N)
                im = ax.imshow(mode, cmap="RdBu_r")
                ax.set_title(f"Mode {i + 1}")
                plt.colorbar(im, ax=ax)
        fig.suptitle("First 4 POD Modes")
        fig.tight_layout()
        fig.savefig(str(FIGURES_DIR / "podqr_modes.png"), dpi=100)
        plt.close(fig)
        print("Saved figures/podqr_modes.png")
    except Exception as e:
        print(f"Warning: could not save mode figure: {e}")
