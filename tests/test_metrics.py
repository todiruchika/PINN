import numpy as np
import pytest
from src.eval.metrics import all_metrics, relative_l2, mae, rmse, mse, peak_temp_error, hotspot_localization_error


def test_relative_l2_zero():
    a = np.ones((4, 4))
    assert relative_l2(a, a) == pytest.approx(0.0)


def test_mae_known():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    true = np.array([[0.0, 0.0], [0.0, 0.0]])
    assert mae(pred, true) == pytest.approx(2.5)


def test_rmse_known():
    pred = np.array([[3.0]])
    true = np.array([[0.0]])
    assert rmse(pred, true) == pytest.approx(3.0)


def test_mse_known():
    pred = np.array([[3.0]])
    true = np.array([[0.0]])
    assert mse(pred, true) == pytest.approx(9.0)


def test_peak_temp_error():
    pred = np.array([[5.0, 2.0]])
    true = np.array([[3.0, 2.0]])
    assert peak_temp_error(pred, true) == pytest.approx(2.0)


def test_hotspot_localization():
    pred = np.zeros((4, 4))
    pred[0, 0] = 10.0
    true = np.zeros((4, 4))
    true[0, 2] = 10.0
    assert hotspot_localization_error(pred, true) == pytest.approx(2.0)


def test_hotspot_localization_ignores_masked_nodes():
    pred = np.zeros((2, 3))
    true = np.zeros((2, 3))
    pred[0, 0] = 100.0
    pred[1, 1] = 10.0
    true[0, 2] = 100.0
    true[1, 1] = 10.0
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    mask[0, 2] = False
    assert hotspot_localization_error(pred, true, mask) == pytest.approx(0.0)
    assert all_metrics(pred, true, mask)["hotspot_localization_cells"] == pytest.approx(0.0)


def test_masked_mae():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    true = np.array([[0.0, 0.0], [0.0, 0.0]])
    mask = np.array([[True, False], [True, False]])
    assert mae(pred, true, mask) == pytest.approx(2.0)
