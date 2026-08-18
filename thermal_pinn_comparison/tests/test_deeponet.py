"""Sanity checks for Model A: shapes, determinism of the data path, one tiny
end-to-end training step. Not a substitute for a full training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import GlobalScaler, chip_ids, default_data_root, grid_points, load_chip  # noqa: E402
from models.deeponet_baseline import build_deeponet  # noqa: E402


def test_model_forward_shapes():
    n_sensors = 5
    model = build_deeponet({"n_sensors": n_sensors, "net": {"p": 16, "branch_width": 16,
                                                              "trunk_width": 16, "branch_depth": 2,
                                                              "trunk_depth": 2}})
    batch, n_query = 4, 10
    branch_in = torch.randn(batch, 3 * n_sensors)
    xy = torch.rand(n_query, 2)
    out = model(branch_in, xy)
    assert out.shape == (batch, n_query)

    single = torch.randn(3 * n_sensors)
    grid_xy = torch.tensor(grid_points(8), dtype=torch.float32)
    field = model.predict_field(single, grid_xy)
    assert field.shape == (64,)


def test_data_loader_matches_verified_facts():
    root = default_data_root()
    ids = chip_ids(root, "train")
    assert len(ids) == 500
    chip = load_chip(ids[0], "train", root)
    pos, temps = chip.sensors(5)
    assert pos.shape == (5, 2)
    assert temps.shape == (5,)
    field = chip.field()
    assert field.shape == (128, 128)
    # boundary equals T_boundary, per the verified dataset facts in the README
    border = np.concatenate([field[0, :], field[-1, :], field[1:-1, 0], field[1:-1, -1]])
    assert np.allclose(border, chip.T_boundary, atol=1e-2)


def test_global_scaler_roundtrip():
    root = default_data_root()
    scaler = GlobalScaler.fit(root, split="train")
    x = np.array([50.0, 1000.0, 4000.0])
    assert np.allclose(scaler.denormalise(scaler.normalise(x)), x, atol=1e-6)


def test_one_training_step_reduces_loss_on_tiny_batch():
    root = default_data_root()
    ids = chip_ids(root, "train")[:8]
    n_sensors = 5
    scaler = GlobalScaler.fit(root, split="train")

    branch, field = [], []
    for sid in ids:
        chip = load_chip(sid, "train", root)
        pos, temps = chip.sensors(n_sensors)
        t_norm = scaler.normalise(temps)
        branch.append(np.concatenate([pos, t_norm.reshape(-1, 1)], axis=1).reshape(-1))
        field.append(scaler.normalise(chip.field()).reshape(-1))
    branch_t = torch.tensor(np.stack(branch), dtype=torch.float32)
    field_t = torch.tensor(np.stack(field), dtype=torch.float32)
    grid_xy = torch.tensor(grid_points(128), dtype=torch.float32)

    model = build_deeponet({"n_sensors": n_sensors, "net": {"p": 32, "branch_width": 32, "trunk_width": 32}})
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    q_idx = torch.randint(0, grid_xy.shape[0], (256,))
    xy_q = grid_xy[q_idx]
    targets = field_t[:, q_idx]

    losses = []
    for _ in range(20):
        pred = model(branch_t, xy_q)
        loss = torch.mean((pred - targets) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


if __name__ == "__main__":
    test_model_forward_shapes()
    test_data_loader_matches_verified_facts()
    test_global_scaler_roundtrip()
    test_one_training_step_reduces_loss_on_tiny_batch()
    print("all tests passed")
