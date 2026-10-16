"""Build the FNO/PINO thesis figures and summary table from saved results.

Reads results/<run>/N<n>_seed<s>/results.json and writes to results/thesis/:
    summary.md / summary.csv      one row per (variant, sensor count)
    error_vs_sensors.png          test relative L2 error against sensor count
    training_curves.png           validation error against training step

Usage (from the repo root):
    python -m src.eval.thesis_figures
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/thesis")

# Variant label -> results folder. Each folder holds N<n>_seed<s>/results.json.
VARIANTS = {
    "baseline (source w 1.0)": Path("results/fno_pino"),
    "source w 0.1": Path("results/fno_pino_wsrc0.1"),
    "source w 0.1 + hot-pixel w 4": Path("results/fno_pino_hotw4"),
}
# Single-run ablations at 10 sensors (seed 0). Shown in the table only.
ABLATIONS = {
    "physics weight 0.1": Path("results/fno_pino_wphys0.1"),
    "16 Fourier modes": Path("results/fno_pino_modes16"),
    "10000 steps (source w 1.0)": Path("results/fno_pino_10k"),
}
SENSOR_COUNTS = [5, 8, 10]
METRICS = [
    ("relative_l2", "relative L2 error (%)", 100.0),
    ("mae_C", "MAE (°C)", 1.0),
    ("peak_temp_error_C", "peak temperature error (°C)", 1.0),
    ("hotspot_localization_cells", "hotspot location error (cells)", 1.0),
]


def load_runs(folder: Path, n_sensors: int) -> list[dict]:
    """Every seed available for one variant and sensor count."""
    runs = []
    for path in sorted(folder.glob(f"N{n_sensors}_seed*/results.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def summarise(runs: list[dict], metric: str) -> tuple[float, float, int]:
    values = [run["test"][metric] for run in runs]
    spread = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), spread, len(values)


def write_summary(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "| Variant | Sensors | Seeds | Relative L2 | MAE (°C) | Peak error (°C) | Hotspot error (cells) | Train steps |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['sensors']} | {row['seeds']} "
            f"| {row['relative_l2_pct']} | {row['mae_C']} | {row['peak_temp_error_C']} "
            f"| {row['hotspot_localization_cells']} | {row['steps']} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_error_vs_sensors(table: dict) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, by_n in table.items():
        xs = [n for n in SENSOR_COUNTS if n in by_n]
        means = [by_n[n]["relative_l2"][0] * 100 for n in xs]
        spreads = [by_n[n]["relative_l2"][1] * 100 for n in xs]
        ax.errorbar(xs, means, yerr=spreads, marker="o", capsize=3, label=label)
    ax.set_xlabel("number of sensors")
    ax.set_ylabel("test relative L2 error (%)")
    ax.set_xticks(SENSOR_COUNTS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("FNO/PINO: error against sensor count\n(bars: std over seeds where more than one seed exists)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_vs_sensors.png", dpi=150)
    plt.close(fig)


def plot_training_curves() -> None:
    colours = {label: f"C{i}" for i, label in enumerate(VARIANTS)}   # same colour for a variant in every panel
    fig, axes = plt.subplots(1, len(SENSOR_COUNTS), figsize=(12, 3.8), sharey=True)
    for ax, n in zip(axes, SENSOR_COUNTS):
        for label, folder in VARIANTS.items():
            for run in load_runs(folder, n)[:1]:   # seed 0 only, to keep the figure readable
                steps = [h["step"] for h in run["history"]]
                errors = [h["val_relative_l2"] * 100 for h in run["history"]]
                ax.plot(steps, errors, marker=".", color=colours[label], label=label)
        ax.set_title(f"{n} sensors (seed 0)", fontsize=10)
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
        if ax.get_lines():
            ax.legend(fontsize=7)
    axes[0].set_ylabel("validation relative L2 (%)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows, table = [], {}
    for label, folder in {**VARIANTS, **ABLATIONS}.items():
        table.setdefault(label, {})
        for n in SENSOR_COUNTS:
            runs = load_runs(folder, n)
            if not runs:
                continue
            per_metric = {}
            for metric, _, scale in METRICS:
                per_metric[metric] = summarise(runs, metric)
            table[label][n] = {m: (v[0], v[1]) for m, v in per_metric.items()}
            rows.append({
                "variant": label,
                "sensors": n,
                "seeds": len(runs),
                "relative_l2_pct": f"{per_metric['relative_l2'][0] * 100:.2f} ± {per_metric['relative_l2'][1] * 100:.2f}",
                "mae_C": f"{per_metric['mae_C'][0]:.2f} ± {per_metric['mae_C'][1]:.2f}",
                "peak_temp_error_C": f"{per_metric['peak_temp_error_C'][0]:.2f} ± {per_metric['peak_temp_error_C'][1]:.2f}",
                "hotspot_localization_cells": f"{per_metric['hotspot_localization_cells'][0]:.1f} ± {per_metric['hotspot_localization_cells'][1]:.1f}",
                "steps": runs[0]["steps"],
            })
    if not rows:
        raise SystemExit("no results found")
    write_summary(rows)
    plot_error_vs_sensors({label: table[label] for label in VARIANTS})
    plot_training_curves()
    print(f"wrote {len(rows)} summary rows and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
