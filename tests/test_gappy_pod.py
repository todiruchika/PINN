import numpy as np
import pytest
from pathlib import Path
from src.eval.gappy_pod import gappy_pod_predict
from src.sensors.podqr import fit_podqr_layout
from src.data.splits import load_splits


def ensure_basis():
    if not Path("sensors/layouts/podqr_basis.npz").exists():
        splits = load_splits()
        fit_podqr_layout(splits["train"])


def test_gappy_pod_finite():
    ensure_basis()
    splits = load_splits()
    sample_id = splits["train"][0]
    pred, true = gappy_pod_predict(sample_id, n_sensors=10, use_generated=False)
    assert np.isfinite(pred).all(), "Reconstruction contains NaN/Inf"
    assert pred.shape == (128, 128)


def test_gappy_pod_rmse():
    ensure_basis()
    splits = load_splits()
    sample_id = splits["train"][0]
    pred, true = gappy_pod_predict(sample_id, n_sensors=10, use_generated=False)
    err = float(np.sqrt(np.mean((pred - true) ** 2)))
    print(f"\nGappy POD RMSE on {sample_id} (N=10): {err:.4f} degC")
    assert err < 100.0, f"RMSE {err:.2f} degC > 100 — reconstruction likely wrong"
