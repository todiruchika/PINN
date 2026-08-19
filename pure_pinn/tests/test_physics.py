"""Sanity checks on the shared physics code, before Model C reuses it.

Run with:  python tests/test_physics.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import ChipScaler, grid_coords, load_chip  # noqa: E402
from losses import laplacian, pde_residual, sample_boundary  # noqa: E402
from models.pinn import TemperatureField, distance_function  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_laplacian_analytic() -> None:
    """d2/dx2 + d2/dy2 of sin(pi x) sin(pi y) is -2 pi^2 times itself."""
    xy = torch.rand(256, 2, dtype=torch.float64)
    fn = lambda p: torch.sin(np.pi * p[:, 0:1]) * torch.sin(np.pi * p[:, 1:2])
    value, lap = laplacian(fn, xy)
    expected = -2 * np.pi**2 * value
    err = float(torch.max(torch.abs(lap - expected)))
    check("laplacian matches analytic result", err < 1e-8, f"max err {err:.2e}")


def test_manufactured_solution() -> None:
    """The residual vanishes for a matching (T, f) pair: lap T + f = 0."""
    xy = torch.rand(256, 2, dtype=torch.float64)
    T = lambda p: torch.sin(np.pi * p[:, 0:1]) * torch.sin(np.pi * p[:, 1:2])
    f = lambda p: 2 * np.pi**2 * torch.sin(np.pi * p[:, 0:1]) * torch.sin(np.pi * p[:, 1:2])
    res = float(torch.max(torch.abs(pde_residual(T, f, xy))))
    check("residual vanishes on a manufactured solution", res < 1e-8, f"max |res| {res:.2e}")


def test_hard_bc_exact() -> None:
    """The hard-BC construction is exactly 0 (normalised) on dOmega, for any weights."""
    torch.manual_seed(0)
    field = TemperatureField(bc_mode="hard", width=32, depth=3)
    xy_b = sample_boundary(2000, torch.Generator().manual_seed(1), dtype=torch.float32)
    with torch.no_grad():
        vals = field(xy_b).abs().max().item()
        d_max = distance_function(torch.tensor([[0.5, 0.5]])).item()
    check("hard BC exact on dOmega", vals < 1e-6, f"max |T_norm| {vals:.2e}")
    check("distance function peaks at 1", abs(d_max - 1.0) < 1e-6, f"d(0.5,0.5)={d_max:.4f}")


def test_dataset_pde_consistency() -> None:
    """The dataset's own fields satisfy lap T_norm + f_norm = 0 in our normalisation.

    Finite differences on a 128-grid over staircased rectangle edges leave a
    few percent of relative error, so the tolerance is loose by design; the
    point is that the sign, the k-scaling and the normalisation all line up.
    """
    chip = load_chip("thermal_00001", "train")
    T = chip.field()
    scaler = ChipScaler(offset=chip.T_boundary, scale=float(chip.peak_temperature - chip.T_boundary))
    Tn = scaler.normalise(T)
    f_norm = chip.source_field(T.shape[0]) * scaler.source_scale(chip.k)

    n = T.shape[0]
    h = 1.0 / (n - 1)
    lap = (Tn[2:, 1:-1] + Tn[:-2, 1:-1] + Tn[1:-1, 2:] + Tn[1:-1, :-2] - 4 * Tn[1:-1, 1:-1]) / h**2
    res = lap + f_norm[1:-1, 1:-1]
    rel = float(np.abs(res).mean() / np.abs(f_norm[1:-1, 1:-1]).mean())
    check("dataset satisfies our normalised PDE", rel < 0.10, f"mean rel residual {rel:.3f}")

    border = np.concatenate([T[0, :], T[-1, :], T[1:-1, 0], T[1:-1, -1]])
    dev = float(np.max(np.abs(border - chip.T_boundary)))
    check("dataset border equals T_boundary", dev < 1e-3, f"max dev {dev:.2e} degC")


def test_sensor_grid_alignment() -> None:
    """Sensor readings equal the matrix value at round(coord * 127), row = y."""
    chip = load_chip("thermal_00001", "train")
    T = chip.field()
    n = T.shape[0]
    worst = 0.0
    for n_sensors in (5, 8, 10):
        pos, temps = chip.sensors(n_sensors)
        j = np.rint(pos[:, 0] * (n - 1)).astype(int)
        i = np.rint(pos[:, 1] * (n - 1)).astype(int)
        worst = max(worst, float(np.max(np.abs(T[i, j] - temps))))
    check("sensors align with matrix indexing (row=y, col=x)", worst < 1e-2, f"max dev {worst:.2e} degC")


def test_grid_orientation() -> None:
    X, Y = grid_coords(4)
    check("grid_coords orientation", np.allclose(X[0], [0, 1 / 3, 2 / 3, 1]) and np.allclose(Y[:, 0], [0, 1 / 3, 2 / 3, 1]))


if __name__ == "__main__":
    test_laplacian_analytic()
    test_manufactured_solution()
    test_hard_bc_exact()
    test_dataset_pde_consistency()
    test_sensor_grid_alignment()
    test_grid_orientation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("all physics checks passed")
