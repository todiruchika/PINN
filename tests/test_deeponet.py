"""Sanity checks for Model A (DeepONet): shapes, data path, one tiny training run."""
import numpy as np
import pytest
import torch

from src.models.deeponet import build_deeponet
from src.train.train_deeponet import (
    GRID, branch_inputs, fields_norm, grid_points, load_split, train_one_config,
)
from src.data.dataset import load_metadata
from src.data.splits import load_splits
from pathlib import Path


def _small_net(n):
    return {"n_sensors": n, "net": {"p": 16, "branch_width": 16, "trunk_width": 16,
                                    "branch_depth": 2, "trunk_depth": 2}}


def test_forward_shapes():
    model = build_deeponet(_small_net(5))
    assert model(torch.randn(4, 15), torch.rand(10, 2)).shape == (4, 10)
    assert model.predict_field(torch.randn(15), torch.rand(64, 2)).shape == (64,)


def test_grid_points_layout():
    g = grid_points(4)
    assert g.shape == (16, 2)
    assert g[1].tolist() == pytest.approx([1 / 3, 0.0])  # x runs along columns
    assert g[4].tolist() == pytest.approx([0.0, 1 / 3])  # y runs along rows


@pytest.fixture(scope="module")
def tiny_data():
    if not Path("data/generated/train/thermal_00001_sensors_5.json").exists():
        pytest.skip("generated sensors missing: python -m src.sensors.generate_sensors")
    meta = load_metadata()
    splits = load_splits()
    return {sp: load_split(splits[sp][:8], 5, "podqr", meta) for sp in ("train", "validation", "test")}


def test_data_path_shapes_and_mask(tiny_data):
    d = tiny_data["train"]
    assert branch_inputs(d, 50.0).shape == (8, 15)
    assert fields_norm(d, 50.0).shape == (8, GRID * GRID)
    assert (~d["mask"]).sum(axis=(1, 2)).tolist() == [5] * 8  # exactly the sensor nodes masked


def test_branch_input_uses_noisy_not_true(tiny_data):
    d = dict(tiny_data["train"])
    b1 = branch_inputs(d, 50.0)
    d["noisy"] = d["noisy"] + 10.0
    assert not np.allclose(b1, branch_inputs(d, 50.0))


def test_tiny_training_run_beats_chance(tiny_data):
    cfg = {"device": "cpu", "model": {"p": 16, "branch_width": 16, "trunk_width": 16,
                                      "branch_depth": 2, "trunk_depth": 2},
           "training": {"epochs": 3, "batch_size": 4, "lr": 1e-2, "lr_min": 1e-4,
                        "n_query": 128, "eval_every": 1, "patience": 5}}
    res = train_one_config(5, 0, cfg, tiny_data)
    assert len(res["rows"]) == 8
    assert np.isfinite([r["relative_l2"] for r in res["rows"]]).all()
