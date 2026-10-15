"""Shared metrics and results bookkeeping for all three models.

Metrics follow ``PROTOCOL.md``: every field error is computed on the
**non-sensor nodes only** (the sensor pixels of the chip's N-sensor layout are
masked out), and reported in degC.

primary       relative L2 field error
secondary     MAE, RMSE, MSE, and max absolute error (hotspot misses matter more
              than mean error)
peak          peak-temperature error and hotspot localisation (grid cells)
plus          boundary flux (does the adiabatic edge hold?), sensor fit, model
              size, and -- for Model B only -- how well the inferred source
              recovers the ground-truth blocks.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import grid_coords
from utils import resolve_path

METRIC_FIELDS = [
    # run identification
    "model", "bc_mode", "source_kind", "split", "sample_id", "n_sensors", "seed",
    # field accuracy (non-sensor nodes)
    "rel_l2", "rmse", "mae", "mse", "max_abs_err",
    # structure (non-sensor nodes)
    "peak_temp_err", "hotspot_loc_cells", "pred_peak_temp", "true_peak_temp",
    # constraint satisfaction / fit
    "boundary_flux_mean", "sensor_rmse",
    # inferred source (Model B)
    "source_rel_l2", "source_centre_err",
    # cost
    "n_params", "train_seconds", "inference_seconds", "final_loss", "final_residual", "final_data_loss",
]


# --------------------------------------------------------------------------- #
# field metrics
# --------------------------------------------------------------------------- #
def sensor_node_mask(positions: np.ndarray, n: int = 128) -> np.ndarray:
    """Boolean ``(n, n)`` mask, ``True`` on the sensor pixels (row = y, col = x)."""
    mask = np.zeros((n, n), dtype=bool)
    j = np.rint(positions[:, 0] * (n - 1)).astype(int)
    i = np.rint(positions[:, 1] * (n - 1)).astype(int)
    mask[i, j] = True
    return mask


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """``||pred - true||_2 / ||true||_2`` over the given values."""
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def _peak_cell(field: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    """``(row, col)`` of the maximum over the ``valid`` nodes."""
    masked = np.where(valid, field, -np.inf)
    i, j = np.unravel_index(int(np.argmax(masked)), field.shape)
    return int(i), int(j)


def boundary_flux_mean(field: np.ndarray, L: float = 0.012) -> float:
    """Mean ``|dT/dn|`` on the four edges (K/m), by one-sided differences.

    The adiabatic condition says this is 0. The dataset itself is not exactly
    0 (stride-downsampling artifact, see ``PROTOCOL.md``), so compare against
    the same number computed on the ground truth.
    """
    h = L / (field.shape[0] - 1)
    edges = [(field[:, 1] - field[:, 0]) / h, (field[:, -1] - field[:, -2]) / h,
             (field[1, :] - field[0, :]) / h, (field[-1, :] - field[-2, :]) / h]
    return float(np.mean(np.abs(np.concatenate(edges))))


def field_metrics(pred: np.ndarray, true: np.ndarray, sensor_mask: np.ndarray, L: float = 0.012) -> dict:
    """Field-level metrics in degC over the non-sensor nodes."""
    if pred.shape != true.shape or sensor_mask.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape} vs mask {sensor_mask.shape}")
    valid = ~sensor_mask
    err = (pred - true)[valid]
    pi, pj = _peak_cell(pred, valid)
    ti, tj = _peak_cell(true, valid)
    return {
        "rel_l2": relative_l2(pred[valid], true[valid]),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err**2)),
        "max_abs_err": float(np.max(np.abs(err))),
        "peak_temp_err": float(abs(pred[pi, pj] - true[ti, tj])),
        "hotspot_loc_cells": float(np.hypot(pi - ti, pj - tj)),
        "pred_peak_temp": float(pred[pi, pj]),
        "true_peak_temp": float(true[ti, tj]),
        "boundary_flux_mean": boundary_flux_mean(pred, L),
    }


def sensor_rmse(pred_field: np.ndarray, positions: np.ndarray, temps: np.ndarray) -> float:
    """RMSE at the sensor pixels -- how well the model honoured its own data."""
    n = pred_field.shape[0]
    j = np.rint(positions[:, 0] * (n - 1)).astype(int)
    i = np.rint(positions[:, 1] * (n - 1)).astype(int)
    return float(np.sqrt(np.mean((pred_field[i, j] - temps) ** 2)))


# --------------------------------------------------------------------------- #
# inferred-source metrics (Model B)
# --------------------------------------------------------------------------- #
def source_metrics(pred_Q: np.ndarray, true_Q: np.ndarray,
                   pred_centres: np.ndarray | None = None,
                   pred_weights: np.ndarray | None = None,
                   true_centres: np.ndarray | None = None,
                   true_weights: np.ndarray | None = None) -> dict:
    """Compare an inferred source field with the ground-truth one.

    ``source_rel_l2`` is the relative L2 error of the rasterised field.
    ``source_centre_err`` greedily matches the strongest predicted blobs to the
    true blocks (by power) and averages the centre distances, so it
    stays meaningful when the two counts differ.
    """
    out = {"source_rel_l2": relative_l2(pred_Q, true_Q) if np.linalg.norm(true_Q) > 0 else float("nan"),
           "source_centre_err": float("nan")}
    if pred_centres is None or true_centres is None or len(true_centres) == 0 or len(pred_centres) == 0:
        return out

    pred_centres = np.asarray(pred_centres, dtype=np.float64)
    true_centres = np.asarray(true_centres, dtype=np.float64)
    pw = np.ones(len(pred_centres)) if pred_weights is None else np.asarray(pred_weights, dtype=np.float64)
    tw = np.ones(len(true_centres)) if true_weights is None else np.asarray(true_weights, dtype=np.float64)

    # rank both sets by power and match greedily, strongest true hotspot first
    true_order = np.argsort(-tw)
    available = list(np.argsort(-pw))
    dists = []
    for t in true_order:
        if not available:
            break
        d = [float(np.hypot(*(pred_centres[p] - true_centres[t]))) for p in available]
        best = int(np.argmin(d))
        dists.append(d[best])
        available.pop(best)
    out["source_centre_err"] = float(np.mean(dists)) if dists else float("nan")
    return out


# --------------------------------------------------------------------------- #
# results file
# --------------------------------------------------------------------------- #
def append_metrics_row(row: dict, path: str | Path = "results/metrics.csv") -> Path:
    """Append one run to ``metrics.csv``, writing the header if the file is new."""
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    clean = {key: row.get(key, "") for key in METRIC_FIELDS}
    extra = {k: v for k, v in row.items() if k not in METRIC_FIELDS}
    if extra:
        raise ValueError(f"unknown metric fields {sorted(extra)}; add them to METRIC_FIELDS")
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(clean)
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def merge_parts(parts_dir: str | Path = "results/parts", out: str | Path = "results/metrics.csv") -> Path:
    """Concatenate per-process metric files into one ``metrics.csv``.

    Parallel runs write one CSV each: concurrent appends to a shared file are
    not atomic, so they are merged after the fact instead.
    """
    parts = sorted(resolve_path(parts_dir).glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"no CSV parts in {resolve_path(parts_dir)}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = df.drop_duplicates(subset=["model", "bc_mode", "source_kind", "split",
                                    "sample_id", "n_sensors", "seed"], keep="last")
    out_path = resolve_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"merged {len(parts)} part file(s) -> {out_path} ({len(df)} rows)")
    return out_path


def comparison_table(path: str | Path = "results/metrics.csv", metric: str = "rel_l2") -> pd.DataFrame:
    """Rows = model, columns = sensor count, cells = ``mean +/- std`` across seeds.

    Per-chip runs (Model B) are first averaged within a seed, so the reported
    std is seed variance -- the quantity the survey flags as under-reported --
    and not chip-to-chip spread.
    """
    df = pd.read_csv(resolve_path(path))
    per_seed = df.groupby(["model", "n_sensors", "seed"])[metric].mean().reset_index()
    agg = per_seed.groupby(["model", "n_sensors"])[metric].agg(["mean", "std", "count"])
    cells = agg.apply(
        lambda r: f"{r['mean']:.4f} +/- {0.0 if pd.isna(r['std']) else r['std']:.4f} (n={int(r['count'])})",
        axis=1,
    )
    return cells.unstack("n_sensors")


def per_chip_summary(path: str | Path = "results/metrics.csv", metric: str = "rel_l2") -> pd.DataFrame:
    """Distribution of ``metric`` across chips -- the right view for Model B."""
    df = pd.read_csv(resolve_path(path))
    return df.groupby(["model", "n_sensors"])[metric].describe()


def plot_error_vs_sensors(path: str | Path = "results/metrics.csv", metric: str = "rel_l2",
                          out: str | Path = "results/figures/error_vs_sensors.png") -> Path:
    """The headline figure: ``metric`` against sensor count, one line per model.

    Error bars are seed std (per-chip runs averaged within a seed first), which
    is the variance the survey flags as under-reported.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(resolve_path(path))
    per_seed = df.groupby(["model", "n_sensors", "seed"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for model, grp in per_seed.groupby("model"):
        agg = grp.groupby("n_sensors")[metric].agg(["mean", "std"]).sort_index()
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"].fillna(0.0),
                    marker="o", capsize=4, label=f"Model {model}")
    ax.set_xlabel("number of sensors N")
    ax.set_ylabel(metric)
    ax.set_xticks(sorted(per_seed["n_sensors"].unique()))
    ax.set_title(f"{metric} vs sensor count (mean +/- seed std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = resolve_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise results/metrics.csv")
    ap.add_argument("--metrics", default="results/metrics.csv")
    ap.add_argument("--metric", default="rel_l2")
    ap.add_argument("--plot", action="store_true", help="also write the error-vs-N figure")
    ap.add_argument("--merge", default=None, help="merge per-process CSVs from this dir first")
    args = ap.parse_args()

    if args.merge:
        merge_parts(args.merge, args.metrics)

    print(f"\n=== comparison table ({args.metric}, mean +/- std over seeds) ===")
    print(comparison_table(args.metrics, args.metric).to_string())
    print(f"\n=== per-chip distribution ({args.metric}) ===")
    print(per_chip_summary(args.metrics, args.metric).to_string())
    if args.plot:
        print(f"\nwrote {plot_error_vs_sensors(args.metrics, args.metric)}")


if __name__ == "__main__":
    main()
