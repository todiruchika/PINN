"""New models on the real thermal maps (FBPINN, FNO): no data leakage, and the model pieces behave as stated."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.real_thermal import DATA_DIR

needs_data = pytest.mark.skipif(not (DATA_DIR / "CPU_i7_git.pkl").exists(), reason="real thermal data not downloaded")


@needs_data
@pytest.mark.parametrize("proc", ["i5", "i7"])
def test_no_test_or_validation_workload_in_training(proc):
    from src.eval.real_hybrid_pinn import restrict_training
    from src.train.train_real_deeponet import prepare

    cfg = yaml.safe_load(open("configs/real_deeponet.yaml"))
    D = prepare(proc, cfg)
    wl = {k: {w for w, _ in v} for k, v in D["frames"].items()}
    assert not wl["train"] & wl["test"] and not wl["train"] & wl["validation"] and not wl["validation"] & wl["test"]
    if proc == "i7":
        for k in (2, 4):
            Dk = prepare(proc, cfg)
            chosen = restrict_training(Dk, k)
            assert len(chosen) == k and set(chosen) <= wl["train"]
            assert len(Dk["maps"]["train"]) == 100 * k                 # only the chosen workloads remain
            assert np.isclose(Dk["mu"], Dk["maps"]["train"].mean())     # normalisation from the kept training maps only


@needs_data
def test_fbpinn_chip_sees_noisy_readings_only():
    from src.eval.real_pinn_runs import RealChip
    from src.train.train_real_deeponet import prepare
    from src.train.train_real_fno import split_readings

    cfg = yaml.safe_load(open("configs/real_deeponet.yaml"))
    D = prepare("i5", cfg)
    r = split_readings(D, 5, cfg)["test"]
    true = D["maps"]["test"][:, D["rc"][:5, 0], D["rc"][:5, 1]]
    assert np.allclose(r, np.round(r)) and not np.allclose(r, true)    # quantised and noisy, not the clean values
    chip = RealChip("x", "test", D["maps"]["test"][0], np.zeros((5, 2)), r[0], D["maps"]["train"].min())
    assert np.array_equal(chip.sensors(5)[1], r[0])


def test_fbpinn_partition_of_unity_and_adiabatic_edges():
    sys.path.insert(0, str(Path("pure_pinn/src").resolve()))
    from models.pinn import MLP
    from src.models.real_fbpinn import FBPINNField

    torch.manual_seed(0)
    f = FBPINNField(MLP)
    xy = torch.rand(200, 2)
    c = torch.cos(math.pi * xy)
    wx, wy = f._axis_windows(c[:, 0]), f._axis_windows(c[:, 1])
    assert torch.allclose(wx.sum(1), torch.ones(200)) and torch.allclose(wy.sum(1), torch.ones(200))
    edge = torch.tensor([[0.0, 0.3], [1.0, 0.7], [0.4, 0.0], [0.6, 1.0]], requires_grad=True)
    g, = torch.autograd.grad(f(edge).sum(), edge)
    assert abs(g[0, 0]) < 1e-5 and abs(g[1, 0]) < 1e-5 and abs(g[2, 1]) < 1e-5 and abs(g[3, 1]) < 1e-5


def test_fno_grid_input_encoding():
    from src.models.real_fno import RealFNO, grid_inputs

    rc = np.array([[3, 4], [10, 2], [7, 7]])
    x = grid_inputs(np.array([[70.0, 80.0]]), rc, (12, 9), 75.0, 5.0, 2)
    assert x.shape == (1, 4, 12, 9)
    assert x[0, 0, 3, 4] == -1.0 and x[0, 0, 10, 2] == 1.0 and x[0, 1].sum() == 2 and x[0, 0, 7, 7] == 0  # third sensor unused
    assert RealFNO()(torch.tensor(x)).shape == (1, 12, 9)
