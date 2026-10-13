"""Evaluation metrics computed on non-sensor (masked) grid nodes."""
import numpy as np


def _check_shapes(pred: np.ndarray, true: np.ndarray):
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs true {true.shape}")


def relative_l2(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """Relative L2 error ||pred-true||_2 / ||true||_2 on masked nodes."""
    _check_shapes(pred, true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    _check_shapes(pred, true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    _check_shapes(pred, true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    _check_shapes(pred, true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float(np.mean((pred - true) ** 2))


def peak_temp_error(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """Difference in peak temperature (degC)."""
    _check_shapes(pred, true)
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return float(pred.max() - true.max())


def hotspot_localization_error(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """Euclidean distance (in grid cells) between predicted and true hotspot."""
    _check_shapes(pred, true)
    if mask is not None:
        if mask.shape != pred.shape:
            raise ValueError(f"Mask shape mismatch: mask {mask.shape} vs data {pred.shape}")
        if not np.any(mask):
            raise ValueError("Mask must include at least one node")
        pred = np.where(mask, pred, -np.inf)
        true = np.where(mask, true, -np.inf)
    pred_idx = np.unravel_index(np.argmax(pred), pred.shape)
    true_idx = np.unravel_index(np.argmax(true), true.shape)
    return float(np.sqrt((pred_idx[0] - true_idx[0]) ** 2 + (pred_idx[1] - true_idx[1]) ** 2))


def all_metrics(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> dict:
    return {
        "relative_l2": relative_l2(pred, true, mask),
        "mae_C": mae(pred, true, mask),
        "rmse_C": rmse(pred, true, mask),
        "mse_C2": mse(pred, true, mask),
        "peak_temp_error_C": peak_temp_error(pred, true, mask),
        "hotspot_localization_cells": hotspot_localization_error(pred, true, mask),
    }
