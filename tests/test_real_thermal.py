"""Tests for the real-thermal-map pipeline. Synthetic arrays only: no download needed."""
import pickle

import numpy as np
import pytest

from src.data.real_thermal import safe_load, split_frames
from src.sensors.real_podqr import pod_basis, podqr_sensors
from src.train.train_real_deeponet import branch_inputs, read_sensors

CFG = {"sensors": {"noise_sigma_C": 1.0, "quant_step_C": 1.0, "calib_sigma_C": 0.5}}


def _write(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def test_safe_load_accepts_plain_arrays(tmp_path):
    p = tmp_path / "ok.pkl"
    _write(p, {"input": np.zeros((4, 3)), "output": np.ones((4, 5, 6))})
    d = safe_load(p)
    assert d["output"].shape == (4, 5, 6) and d["output"].dtype == np.float32


def test_safe_load_blocks_code_execution(tmp_path):
    marker = tmp_path / "pwned"

    class Evil:
        def __reduce__(self):
            import os
            return (os.system, (f"echo x > {marker}",))

    p = tmp_path / "evil.pkl"
    _write(p, {"input": Evil(), "output": np.ones((1, 2, 2))})
    with pytest.raises(pickle.UnpicklingError):
        safe_load(p)
    assert not marker.exists()


def test_safe_load_rejects_wrong_structure(tmp_path):
    p = tmp_path / "bad.pkl"
    _write(p, {"input": np.zeros(3)})
    with pytest.raises(ValueError):
        safe_load(p)


def _workloads(names, n=100):
    return {w: (np.zeros((n, 2)), np.zeros((n, 4, 4))) for w in names}


def test_multi_workload_split_keeps_workloads_whole():
    wl = _workloads([f"w{i}" for i in range(19)])
    sp = split_frames("i7", wl, seed=0)
    sets = {k: {w for w, _ in v} for k, v in sp.items()}
    assert not (sets["train"] & sets["validation"]) and not (sets["train"] & sets["test"]) and not (sets["validation"] & sets["test"])
    assert (len(sets["train"]), len(sets["validation"]), len(sets["test"])) == (13, 3, 3)
    assert sum(len(v) for v in sp.values()) == 1900


def test_single_recording_split_has_gaps():
    sp = split_frames("tpu", _workloads(["only"]), seed=0)
    tr, va, te = ([i for _, i in sp[k]] for k in ("train", "validation", "test"))
    assert max(tr) + 2 < min(va) and max(va) + 2 < min(te)      # at least 2 unused frames between blocks
    assert (len(tr), len(va), len(te)) == (68, 14, 14)


def test_podqr_sensors_respect_margin_and_are_nested():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(12, 30, 40))
    mean, basis = pod_basis(base, energy=0.5, min_modes=10)
    assert basis.shape[1] >= 10                                  # never fewer modes than the largest sensor count
    rc = podqr_sensors(basis, (30, 40), n_max=10, margin=3)
    assert rc.shape == (10, 2) and len({tuple(x) for x in rc}) == 10
    assert rc[:, 0].min() >= 3 and rc[:, 0].max() <= 26 and rc[:, 1].min() >= 3 and rc[:, 1].max() <= 36
    assert np.array_equal(podqr_sensors(basis, (30, 40), n_max=5, margin=3), rc[:5])


def test_sensor_readings_deterministic_and_branch_shape():
    maps = np.random.default_rng(1).uniform(60, 90, size=(6, 30, 40)).astype(np.float32)
    rc = np.array([[5, 6], [10, 20], [15, 3], [20, 30], [25, 10]])
    a = read_sensors(maps, rc, 0, 5, CFG, 0)
    b = read_sensors(maps, rc, 0, 5, CFG, 0)
    assert np.array_equal(a, b) and a.shape == (6, 5) and np.allclose(a, np.round(a))
    assert not np.array_equal(a, read_sensors(maps, rc, 0, 5, CFG, 1))          # different split = different noise
    assert branch_inputs(a, rc, (30, 40), 75.0, 8.0, 5).shape == (6, 15)
