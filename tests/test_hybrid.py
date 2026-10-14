"""Sanity checks for Model C (physics-informed DeepONet)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import load_metadata
from src.data.splits import load_splits
from src.models.hybrid import HybridDeepONet, build_hybrid
from src.physics.losses import ChipContext, Physics, pde_residual
from src.train.train_deeponet import load_split
from src.train.train_hybrid import train_one_config


def _net(**kw):
    return HybridDeepONet(5, p=16, branch_width=16, trunk_width=16, branch_depth=2, trunk_depth=2, **kw)


def test_forward_shapes_and_source_nonnegative():
    m = _net()
    theta, q = m(torch.randn(4, 15), torch.rand(10, 2))
    assert theta.shape == q.shape == (4, 10)
    assert (q >= 0).all()
    t1, q1 = m.predict_field(torch.randn(15), torch.rand(64, 2))
    assert t1.shape == q1.shape == (64,)


def test_pointwise_matches_batched_forward():
    m = _net()
    b, xy = torch.randn(3, 15), torch.rand(7, 2)
    theta, q = m(b, xy)
    emb = m.embed_branch(b)
    b_rep = emb.repeat_interleave(7, dim=0)
    xy_rep = xy.repeat(3, 1)
    assert torch.allclose(m.theta_pointwise(b_rep, xy_rep).view(3, 7), theta, atol=1e-5)
    assert torch.allclose(m.q_pointwise(b_rep, xy_rep).view(3, 7), q, atol=1e-5)


def test_hard_bc_gives_zero_normal_derivative_on_edges():
    m = _net(hard_bc=True)
    b = torch.randn(1, 15)
    t = torch.linspace(0.1, 0.9, 5)
    for edge in (torch.stack([torch.zeros(5), t], 1), torch.stack([torch.ones(5), t], 1)):   # x = 0, 1
        xy = edge.clone().requires_grad_(True)
        g = torch.autograd.grad(m(b, xy)[0].sum(), xy)[0]
        assert g[:, 0].abs().max() < 1e-5
    for edge in (torch.stack([t, torch.zeros(5)], 1), torch.stack([t, torch.ones(5)], 1)):   # y = 0, 1
        xy = edge.clone().requires_grad_(True)
        g = torch.autograd.grad(m(b, xy)[0].sum(), xy)[0]
        assert g[:, 1].abs().max() < 1e-5


def test_residual_is_per_chip():
    """Chips with different branch embeddings must get different, independent residuals."""
    m = _net()
    phys, nc = Physics(), 6
    b = torch.randn(2, 15)
    emb = m.embed_branch(b).repeat_interleave(nc, 0)
    xy = torch.rand(nc, 2).repeat(2, 1)
    ctx = ChipContext(T_amb=torch.full((2 * nc, 1), 30.0), scale=90.0)
    res = pde_residual(lambda x: m.theta_pointwise(emb, x),
                       lambda x, Tc: m.q_pointwise(emb, x) * phys.leakage_factor(Tc), xy, phys, ctx)
    # chip 0 alone must give the same residual as its rows in the stacked call
    emb0 = m.embed_branch(b[:1]).repeat_interleave(nc, 0)
    ctx0 = ChipContext(T_amb=torch.full((nc, 1), 30.0), scale=90.0)
    res0 = pde_residual(lambda x: m.theta_pointwise(emb0, x),
                        lambda x, Tc: m.q_pointwise(emb0, x) * phys.leakage_factor(Tc), xy[:nc], phys, ctx0)
    assert torch.allclose(res[:nc], res0, atol=1e-5)
    assert not torch.allclose(res[:nc], res[nc:], atol=1e-5)


@pytest.fixture(scope="module")
def tiny_data():
    if not Path("data/generated/train/thermal_00001_sensors_5.json").exists():
        pytest.skip("generated sensors missing")
    meta, splits = load_metadata(), load_splits()
    return {sp: load_split(splits[sp][:8], 5, "podqr", meta) for sp in ("train", "validation", "test")}


def test_tiny_training_run_with_physics(tiny_data):
    cfg = {"device": "cpu",
           "model": {"p": 16, "branch_width": 16, "trunk_width": 16, "branch_depth": 2, "trunk_depth": 2},
           "training": {"epochs": 3, "batch_size": 4, "lr": 1e-2, "lr_min": 1e-4, "n_query": 64,
                        "n_colloc": 16, "lambda_pde": 1.0, "pde_ramp_epochs": 1, "lambda_q": 0.5,
                        "eval_every": 1, "patience": 5}}
    res = train_one_config(5, 0, cfg, tiny_data)
    assert len(res["rows"]) == 8
    assert np.isfinite([r["relative_l2"] for r in res["rows"]]).all()
    assert np.isfinite([r["source_rel_l2"] for r in res["rows"]]).all()


def test_pde_residual_metric_is_small_for_truth_and_large_for_noise():
    from src.physics.residual import relative_pde_residual
    meta = load_metadata()
    row = meta[meta.split == "train"].iloc[0]
    d = f"data/raw/dataset/train/matrices/{row.sample_id}"
    T, Q = np.load(d + ".npy"), np.load(d + "_Q.npy")
    floor = relative_pde_residual(T, Q, row.T_amb_C)
    noisy = relative_pde_residual(T + np.random.default_rng(0).normal(0, 0.5, T.shape), Q, row.T_amb_C)
    assert floor < 0.5 < noisy
