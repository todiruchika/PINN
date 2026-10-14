"""Tests for the data-driven PINN (physics, model, data arrays, training loop)."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.arrays import load_split_arrays
from src.models.conditional_pinn import ConditionalPINN
from src.physics.autodiff import Physics, pde_residual
from src.train import data_driven as dd

ROOT = Path(__file__).resolve().parents[1]


def _generator_module():
    spec = importlib.util.spec_from_file_location("pde_solver", ROOT / "generator" / "pde_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# physics
# --------------------------------------------------------------------------- #
def test_conductivity_matches_generator():
    T = np.linspace(-40.0, 1000.0, 400)
    ours = Physics().k(torch.tensor(T, dtype=torch.float64)).numpy()
    ref = _generator_module().k_silicon(T)
    assert np.max(np.abs(ours - ref)) < 1e-9


def test_residual_vanishes_on_manufactured_solution():
    """Constant k: theta = cos(pi x) cos(pi y) gives q_tilde = (1 + 2 pi^2 eps k) theta."""
    kc = 110.0
    phys = Physics(k_const=kc)
    coeff = 1.0 + 2 * math.pi**2 * phys.eps * kc

    def fn(xy):
        th = torch.cos(math.pi * xy[:, 0:1]) * torch.cos(math.pi * xy[:, 1:2])
        return th, coeff * th

    xy = torch.rand(256, 2, dtype=torch.float64)
    res = pde_residual(fn, xy, torch.full((256, 1), 30.0, dtype=torch.float64), phys, 50.0)
    assert float(res.abs().max()) < 1e-8


def test_residual_variable_k_matches_finite_differences():
    """Autodiff div(k(T) grad theta) agrees with a central-difference flux divergence."""
    phys, S = Physics(), 50.0
    T_amb = torch.full((64, 1), 35.0, dtype=torch.float64)
    theta = lambda p: 0.8 * torch.sin(2.0 * p[:, 0:1]) * torch.cos(1.5 * p[:, 1:2]) + 0.4
    fn = lambda p: (theta(p), torch.zeros_like(theta(p)))
    xy = 0.2 + 0.6 * torch.rand(64, 2, dtype=torch.float64)
    res = pde_residual(fn, xy, T_amb, phys, S)
    auto_div = -(res - theta(xy)) / phys.eps                              # res = -eps div + theta - 0

    h = 1e-5
    def flux(p, d):
        p = p.clone().requires_grad_(True)
        g = torch.autograd.grad(theta(p).sum(), p)[0][:, d:d + 1]
        return phys.k(T_amb + S * theta(p)).detach() * g.detach()
    fd = torch.zeros_like(auto_div)
    for d in range(2):
        e = torch.zeros(1, 2, dtype=torch.float64); e[0, d] = h
        fd = fd + (flux(xy + e, d) - flux(xy - e, d)) / (2 * h)
    assert float((auto_div - fd).abs().max()) < 1e-4


def _fd_residual_ratio(T, Q, T_amb, peak, phys):
    n = 128
    h = 1.0 / (n - 1)
    S = peak - T_amb
    th = (T - T_amb) / S
    k = phys.k(torch.tensor(T)).numpy()
    kE = 2 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    kN = 2 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])
    fx, fy = kE * (th[:, 1:] - th[:, :-1]) / h, kN * (th[1:, :] - th[:-1, :]) / h
    I = slice(1, -1)
    div = (fx[I, 1:] - fx[I, :-1]) / h + (fy[1:, I] - fy[:-1, I]) / h
    q = Q / (phys.h_eff * S)
    res = -phys.eps * div + th[I, I] - q[I, I]
    return float(np.abs(res).mean() / np.abs(q[I, I]).mean())


def test_dataset_fields_satisfy_the_normalised_pde():
    """The stored fields satisfy the normalised PDE (~7% stencil residual); a wrong t_die does not."""
    ds = load_split_arrays("train", 5)
    peaks = ds.T.reshape(len(ds), -1).max(axis=1)
    good, bad = [], []
    for i in range(6):
        args = (ds.T[i].astype(np.float64), ds.Q[i].astype(np.float64), ds.T_amb[i], float(peaks[i]))
        good.append(_fd_residual_ratio(*args, Physics()))
        bad.append(_fd_residual_ratio(*args, Physics(t_die=1e-3)))
    assert np.median(good) < 0.12
    assert np.median(bad) > 0.5


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def _small(n=5, **kw):
    return ConditionalPINN(n_sensors=n, latent=8, branch_width=16, branch_depth=2,
                           trunk_width=16, trunk_depth=2, **kw)


def test_model_shapes_and_shared_vs_per_chip_points_agree():
    torch.manual_seed(0)
    m = _small()
    B, P = 3, 7
    cond = torch.randn(B, 6)
    xy = torch.rand(P, 2)
    th_s, q_s = m(cond, xy)
    th_p, q_p = m(cond, xy[None].expand(B, P, 2))
    assert th_s.shape == q_s.shape == (B, P)
    assert torch.allclose(th_s, th_p, atol=1e-6) and torch.allclose(q_s, q_p, atol=1e-6)
    assert float(q_s.min()) >= 0.0


def test_rows_fn_matches_forward():
    torch.manual_seed(1)
    m = _small()
    cond, xy = torch.randn(2, 6), torch.rand(2, 5, 2)
    th, q = m(cond, xy)
    coef_rows = m.coefficients(cond).repeat_interleave(5, dim=0)
    th_r, q_r = m.rows_fn(coef_rows)(xy.reshape(-1, 2))
    assert torch.allclose(th.reshape(-1, 1), th_r, atol=1e-6)
    assert torch.allclose(q.reshape(-1, 1), q_r, atol=1e-6)


def _edge_normal_derivative(m):
    cond = torch.randn(2, m.n_sensors + 1)
    t = torch.rand(200)
    zero, one = torch.zeros_like(t), torch.ones_like(t)
    worst = 0.0
    for pts, d in ((torch.stack([zero, t], 1), 0), (torch.stack([one, t], 1), 0),
                   (torch.stack([t, zero], 1), 1), (torch.stack([t, one], 1), 1)):
        pts = pts.clone().requires_grad_(True)
        th, _ = m(cond, pts[None].expand(2, -1, -1))
        g = torch.autograd.grad(th.sum(), pts)[0]
        worst = max(worst, float(g[:, d].abs().max()))
    return worst


def test_hard_neumann_is_exact_and_soft_is_not():
    torch.manual_seed(2)
    assert _edge_normal_derivative(_small(hard_neumann=True)) < 1e-4
    assert _edge_normal_derivative(_small(hard_neumann=False)) > 1e-3


def test_model_config_roundtrip():
    m = _small(n=8)
    m2 = ConditionalPINN.from_config(m.cfg)
    assert m2.n_params == m.n_params and m2.n_sensors == 8


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def test_split_arrays_shapes_and_sensor_noise_model():
    ds = load_split_arrays("validation", 5)
    n = len(ds)
    assert n == 100 and ds.T.shape == (n, 128, 128) and ds.Q.shape == ds.T.shape
    assert ds.sensors_true.shape == ds.sensors_noisy.shape == (n, 5) and ds.positions_m.shape == (5, 2)
    # stored noisy = true + per-index offset + N(0, 1) then 1 degC quantisation
    resid = ds.sensors_noisy - ds.sensors_true - ds.noise.offsets_C
    expected_std = math.sqrt(ds.noise.sigma_C**2 + ds.noise.quant_step_C**2 / 12)
    assert abs(resid.mean()) < 0.25 and abs(resid.std() - expected_std) < 0.2


def test_sensor_true_values_match_the_fields():
    ds = load_split_arrays("validation", 8)
    cols = np.rint(ds.positions_m[:, 0] / 0.012 * 127).astype(int)
    rows = np.rint(ds.positions_m[:, 1] / 0.012 * 127).astype(int)
    assert np.max(np.abs(ds.T[:, rows, cols] - ds.sensors_true)) < 1e-3


def test_noise_augmentation_is_quantised_and_matches_stored_statistics():
    ds = load_split_arrays("train", 5)
    data = dd.TrainData(ds, Physics(), 50.0, torch.device("cpu"))
    gen = torch.Generator().manual_seed(0)
    idx = torch.arange(len(ds))
    aug = data.sensors(idx, True, gen)
    assert torch.allclose(aug, torch.round(aug))                          # on the 1 degC grid
    d_aug = (aug - data.sensors_true - data.offsets)
    d_stored = (data.sensors_noisy - data.sensors_true - data.offsets)
    assert abs(float(d_aug.std()) - float(d_stored.std())) < 0.1


def test_lr_schedule_endpoints():
    assert dd.lr_at(0, 100, 1e-3, 1e-5) == pytest.approx(1e-3)
    assert dd.lr_at(100, 100, 1e-3, 1e-5) == pytest.approx(1e-5)


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_training_smoke_then_eval_only(tmp_path):
    out = tmp_path / "run"
    base = ["--sensors", "5", "--out", str(out), "--device", "cpu", "--eval-splits", "validation"]
    dd.main(base + ["--steps", "12", "--batch-chips", "4", "--points-data", "64", "--points-phys", "16",
                    "--eval-every", "6", "--eval-chips", "4", "--log-every", "6", "--latent", "8",
                    "--branch-width", "16", "--branch-depth", "2", "--trunk-width", "16", "--trunk-depth", "2"])
    for name in ("best.pt", "last.pt", "history.csv", "seed_0.csv", "per_chip_validation.csv", "summary.json"):
        assert (out / name).exists(), name
    seed = pd.read_csv(out / "seed_0.csv")
    assert seed.loc[0, "config"] == "data_driven_pinn_N5" and np.isfinite(seed.loc[0, "relative_l2"])

    # re-score the saved checkpoint, and resume training from it
    dd.main(base + ["--eval-only"])
    dd.main(base + ["--steps", "18", "--resume", str(out / "last.pt"), "--batch-chips", "4",
                    "--points-data", "64", "--points-phys", "16", "--eval-every", "6", "--eval-chips", "4",
                    "--log-every", "6", "--latent", "8", "--branch-width", "16", "--branch-depth", "2",
                    "--trunk-width", "16", "--trunk-depth", "2"])
    assert int(pd.read_csv(out / "history.csv")["step"].max()) == 18
