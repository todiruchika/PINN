"""Sanity checks on the shared physics code, before Model C reuses it.

Run with:  python tests/test_physics.py   (from anywhere; paths are resolved from the file)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[2] / "generator"))   # the dataset's own solver, for k(T)

from data_loader import ChipScaler, grid_coords, grid_points, load_chip  # noqa: E402
from evaluate import field_metrics, sensor_node_mask  # noqa: E402
from losses import (  # noqa: E402
    ChipContext, Physics, flux_divergence, laplacian, neumann_loss, pde_residual, sample_boundary,
)
from models.pinn import FixedGridSource, TemperatureField  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


# --------------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------------- #
def test_laplacian_analytic() -> None:
    """d2/dx2 + d2/dy2 of sin(pi x) sin(pi y) is -2 pi^2 times itself."""
    xy = torch.rand(256, 2, dtype=torch.float64)
    fn = lambda p: torch.sin(np.pi * p[:, 0:1]) * torch.sin(np.pi * p[:, 1:2])
    value, lap = laplacian(fn, xy)
    err = float(torch.max(torch.abs(lap - (-2 * np.pi**2 * value))))
    check("laplacian matches analytic result", err < 1e-8, f"max err {err:.2e}")


def test_flux_divergence_variable_k() -> None:
    """For theta = x^2 + y^2 and k = theta, div(k grad theta) = 8 theta (incl. the k' |grad|^2 term)."""
    xy = torch.rand(256, 2, dtype=torch.float64)
    fn = lambda p: p[:, 0:1] ** 2 + p[:, 1:2] ** 2
    value, div = flux_divergence(fn, lambda th: th, xy)
    err = float(torch.max(torch.abs(div - 8.0 * value)))
    check("flux divergence handles variable k", err < 1e-8, f"max err {err:.2e}")


def test_manufactured_solution() -> None:
    """With constant k the residual vanishes for a matching (theta, q) pair.

    theta = cos(pi x) cos(pi y) has lap theta = -2 pi^2 theta, so
    -eps k lap theta + theta = (1 + 2 pi^2 eps k) theta =: q_tilde.
    """
    kc = 110.0
    phys = Physics(k_const=kc)
    coeff = 1.0 + 2 * np.pi**2 * phys.eps * kc
    xy = torch.rand(256, 2, dtype=torch.float64)
    theta = lambda p: torch.cos(np.pi * p[:, 0:1]) * torch.cos(np.pi * p[:, 1:2])
    source = lambda p, T: coeff * theta(p)
    res = pde_residual(theta, source, xy, phys, ChipContext(T_amb=30.0, scale=20.0))
    err = float(torch.max(torch.abs(res)))
    check("residual vanishes on a manufactured solution", err < 1e-8, f"max |res| {err:.2e}")


def test_conductivity_matches_generator() -> None:
    """Physics.k reproduces generator/pde_solver.k_silicon, clipping included."""
    from pde_solver import k_silicon

    T = np.linspace(-40.0, 1000.0, 400)
    ours = Physics().k(torch.tensor(T, dtype=torch.float64)).numpy()
    err = float(np.max(np.abs(ours - k_silicon(T))))
    check("k(T) matches generator", err < 1e-9, f"max err {err:.2e} W/(m K)")


# --------------------------------------------------------------------------- #
# boundary condition
# --------------------------------------------------------------------------- #
def test_hard_neumann_exact() -> None:
    """The cosine embedding gives d theta/dn = 0 on dOmega for any weights; a plain MLP does not."""
    torch.manual_seed(0)
    xy_b = sample_boundary(2000, torch.Generator().manual_seed(1), dtype=torch.float32)
    hard = TemperatureField(bc_mode="hard", width=32, depth=3)
    soft = TemperatureField(bc_mode="soft", width=32, depth=3)
    hard_pen = float(neumann_loss(hard, xy_b))
    soft_pen = float(neumann_loss(soft, xy_b))
    check("hard BC: zero normal derivative on dOmega", hard_pen < 1e-10, f"mean (dtheta/dn)^2 {hard_pen:.2e}")
    check("soft BC: plain MLP is not adiabatic (test is meaningful)", soft_pen > 1e-6, f"{soft_pen:.2e}")


# --------------------------------------------------------------------------- #
# dataset consistency
# --------------------------------------------------------------------------- #
def _dataset_residual(chip, physics: Physics) -> float:
    """Mean |R| / mean |q| of the normalised PDE on a ground-truth field (solver-style stencil)."""
    n = 128
    h = 1.0 / (n - 1)
    T, Q = chip.field(), chip.source_field()
    S = chip.peak_temperature - chip.T_amb
    th = (T - chip.T_amb) / S
    k = physics.k(torch.tensor(T)).numpy()
    kE = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])        # harmonic face means, as in the solver
    kN = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    fx = kE * (th[:, 1:] - th[:, :-1]) / h
    fy = kN * (th[1:, :] - th[:-1, :]) / h
    I = slice(1, -1)
    div = (fx[I, 1:] - fx[I, :-1]) / h + (fy[1:, I] - fy[:-1, I]) / h
    q = Q / (physics.h_eff * S)
    res = -physics.eps * div + th[I, I] - q[I, I]
    return float(np.abs(res).mean() / np.abs(q[I, I]).mean())


def test_dataset_pde_consistency() -> None:
    """The dataset's own fields satisfy the normalised PDE, and a wrong constant would not.

    The 128-grid fields come from a 512-grid FEM solve (stride-downsampled) with
    staircased block edges, so a few percent of residual is expected; measured
    median is ~0.07. The diffusion term is about as large as the source, so a
    wrong ``t_die`` or ``h_eff`` pushes this above 1.
    """
    ids = ["thermal_00001", "thermal_00002", "thermal_00003", "thermal_00004", "thermal_00005", "thermal_00006"]
    good = float(np.median([_dataset_residual(load_chip(i, "train"), Physics()) for i in ids]))
    bad = float(np.median([_dataset_residual(load_chip(i, "train"), Physics(t_die=1e-3)) for i in ids]))
    check("dataset satisfies the normalised PDE", good < 0.12, f"median rel residual {good:.3f}")
    check("...and a wrong t_die is detected", bad > 0.5, f"median rel residual {bad:.3f}")


def test_q_integral_matches_blocks() -> None:
    """Stored Q integrates to the energy the generator recorded as input power."""
    import pandas as pd
    chip = load_chip("thermal_00001", "train")
    meta = pd.read_csv(chip.root / "train" / "metadata.csv").set_index("sample_id").loc[chip.sample_id]
    dx = chip.die_L / 127
    total = float(chip.source_field().sum() * dx * dx)
    rel = abs(total / float(meta["energy_bal_P_in"]) - 1.0)
    check("sum(Q) dx dy equals recorded input power", rel < 1e-3, f"relative gap {rel:.2e}")


def test_sensor_grid_alignment() -> None:
    """Sensor ``temperatures_true`` equal the matrix value at round(coord * 127), row = y."""
    chip = load_chip("thermal_00001", "train")
    T = chip.field()
    worst = 0.0
    for n_sensors in (5, 8, 10):
        path = chip.sensor_root / chip.split / f"{chip.sample_id}_sensors_{n_sensors}.json"
        record = json.loads(path.read_text())
        pos, _ = chip.sensors(n_sensors)
        j = np.rint(pos[:, 0] * 127).astype(int)
        i = np.rint(pos[:, 1] * 127).astype(int)
        worst = max(worst, float(np.max(np.abs(T[i, j] - np.asarray(record["temperatures_true"])))))
    check("sensors align with matrix indexing (row=y, col=x)", worst < 1e-3, f"max dev {worst:.2e} degC")


def test_sensor_input_is_noisy_only() -> None:
    """The model-facing readings are the noisy ones (PROTOCOL: temperatures_noisy only)."""
    chip = load_chip("thermal_00001", "train")
    path = chip.sensor_root / chip.split / f"{chip.sample_id}_sensors_5.json"
    record = json.loads(path.read_text())
    _, temps = chip.sensors(5)
    check("sensors() returns temperatures_noisy", np.allclose(temps, record["temperatures_noisy"]))
    check("noisy differs from true", not np.allclose(temps, record["temperatures_true"]))


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def test_grid_orientation() -> None:
    X, Y = grid_coords(4)
    check("grid_coords orientation", np.allclose(X[0], [0, 1 / 3, 2 / 3, 1]) and np.allclose(Y[:, 0], [0, 1 / 3, 2 / 3, 1]))


def test_fixed_grid_source_roundtrip() -> None:
    """The oracle source reproduces its grid exactly at the grid nodes."""
    rng = np.random.default_rng(0)
    grid = rng.uniform(1.0, 5.0, size=(16, 16))
    src = FixedGridSource(grid, q_scale=2.0)
    pts = torch.tensor(grid_points(16), dtype=torch.float32)
    out = src(pts).detach().numpy().reshape(16, 16)
    err = float(np.max(np.abs(out - grid / 2.0)))
    check("FixedGridSource returns grid values at nodes (row=y)", err < 1e-5, f"max err {err:.2e}")


def test_metrics_ignore_sensor_nodes() -> None:
    """An error confined to a sensor pixel does not enter the field metrics."""
    true = np.random.default_rng(1).uniform(30, 80, size=(128, 128))
    pred = true.copy()
    positions = np.array([[0.5, 0.5]])
    mask = sensor_node_mask(positions)
    pred[64, 64] += 500.0                                   # huge error, but only at the sensor node
    m = field_metrics(pred, true, mask)
    check("field metrics exclude sensor nodes", m["max_abs_err"] == 0.0 and m["rel_l2"] == 0.0, f"max_abs {m['max_abs_err']}")
    pred[10, 10] += 5.0
    m2 = field_metrics(pred, true, mask)
    check("field metrics see non-sensor errors", abs(m2["max_abs_err"] - 5.0) < 1e-9)


def test_scaler_uses_ambient() -> None:
    chip = load_chip("thermal_00001", "train")
    _, temps = chip.sensors(5)
    s = ChipScaler.from_sensors(chip, temps)
    check("scaler offset is T_amb", abs(s.offset - chip.T_amb) < 1e-12)
    check("scaler round-trips", np.allclose(s.denormalise(s.normalise(temps)), temps))


if __name__ == "__main__":
    test_laplacian_analytic()
    test_flux_divergence_variable_k()
    test_manufactured_solution()
    test_conductivity_matches_generator()
    test_hard_neumann_exact()
    test_dataset_pde_consistency()
    test_q_integral_matches_blocks()
    test_sensor_grid_alignment()
    test_sensor_input_is_noisy_only()
    test_grid_orientation()
    test_fixed_grid_source_roundtrip()
    test_metrics_ignore_sensor_nodes()
    test_scaler_uses_ambient()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("all physics checks passed")
