"""Shared metrics and results bookkeeping for all three models.

Metrics follow Sec. 6 of the study design:

primary    relative L2 field error over the full held-out image
secondary  max absolute error (hotspot misses matter more than mean error)
localisation  distance between predicted and true peak location
plus        boundary MAE (does the BC actually hold?), sensor fit, and -- for
            Model B only -- how well the inferred source field recovers the
            ground-truth hotspots.
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
    # field accuracy
    "rel_l2", "rmse", "mae", "max_abs_err",
    # structure
    "peak_loc_err", "pred_peak_temp", "true_peak_temp",
    # constraint satisfaction / fit
    "boundary_mae", "sensor_rmse",
    # inferred source (Model B)
    "source_rel_l2", "source_centre_err",
    # cost
    "train_seconds", "inference_seconds", "final_loss", "final_residual", "final_data_loss",
]


# --------------------------------------------------------------------------- #
# field metrics
# --------------------------------------------------------------------------- #
def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    """``||pred - true||_2 / ||true||_2`` over the whole field."""
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


def peak_location(field: np.ndarray) -> tuple[float, float]:
    """``(x, y)`` in ``[0, 1]^2`` of the field's maximum."""
    n = field.shape[0]
    i, j = np.unravel_index(int(np.argmax(field)), field.shape)
    return j / (n - 1), i / (n - 1)


def field_metrics(pred: np.ndarray, true: np.ndarray, T_boundary: float) -> dict:
    """Field-level metrics in physical units (degC)."""
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    err = pred - true
    px, py = peak_location(pred)
    tx, ty = peak_location(true)
    border = np.concatenate([pred[0, :], pred[-1, :], pred[1:-1, 0], pred[1:-1, -1]])
    return {
        "rel_l2": relative_l2(pred, true),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
        "peak_loc_err": float(np.hypot(px - tx, py - ty)),
        "pred_peak_temp": float(np.max(pred)),
        "true_peak_temp": float(np.max(true)),
        "boundary_mae": float(np.mean(np.abs(border - T_boundary))),
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
    true hotspot rectangles (by power) and averages the centre distances, so it
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
