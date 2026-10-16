"""Compare FNO/PINO, FBPINN and XPINN runs (and the teammates' DeepONet) in one table and figure.

Reads results/<run>/N<n>_seed<s>/results.json and writes to results/comparison/:
    summary.md / summary.csv     one row per (model, sensor count)
    error_vs_sensors.png         test relative L2 error against sensor count

Usage (from the repo root):
    python -m src.eval.compare_models
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path("results/comparison")

# Label -> results folder. Missing folders are skipped.
RUNS = {
    "FNO/PINO (final)": Path("results/fno_pino_wsrc0.1"),
    "FNO/PINO + POD estimate": Path("results/fno_pino_podprior"),
    "FBPINN, 3000 steps x 8 chips (full grid)": Path("results/fbpinn"),
    "FBPINN fast (batch 32, 1024 points; 5 sensors: 10k steps, 8 and 10: 6k)": Path("results/fbpinn_fast"),
    "FBPINN fast + POD estimate (10k steps)": Path("results/fbpinn_fast_pod"),
    "FBPINN fast, encoder 256 (10 sensors)": Path("results/fbpinn_fast_enc256"),
    "FBPINN fast, 16 windows (10 sensors)": Path("results/fbpinn_fast_w16"),
    "FBPINN fast, physics off (10 sensors)": Path("results/fbpinn_fast_wphys0"),
    "XPINN fast, physics on, unmasked (first version)": Path("results/xpinn_fast"),
    "XPINN fast, physics on, unmasked + POD estimate": Path("results/xpinn_fast_pod"),
    "XPINN fast, 96 hidden units": Path("results/xpinn_fast_h96"),
    "XPINN fast, physics off": Path("results/xpinn_fast_wphys0"),
    "XPINN fast, physics on, tile-edge nodes masked": Path("results/xpinn_fast_seam"),
    "XPINN fast, patch-averaged physics, tile-edge nodes masked": Path("results/xpinn_fast_seam_pool16"),
}
SENSOR_COUNTS = [5, 8, 10]

# Reference only: DeepONet (branch `smriti`, results/model_a/summary.md), mean of 5 seeds, 400 epochs
# at batch 32. Taken from their summary, not re-run here. Their trainer scores with the same
# src.eval.metrics.all_metrics on non-sensor nodes (checked in src/train/train_deeponet.py), so the
# error definition matches ours.
DEEPONET_REFERENCE = {5: 0.1146, 8: 0.0862, 10: 0.0613}


def load_runs(folder: Path, n_sensors: int) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob(f"N{n_sensors}_seed*/results.json"))]


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for label, folder in RUNS.items():
        for n in SENSOR_COUNTS:
            runs = load_runs(folder, n)
            if not runs:
                continue
            cfg = runs[0]["config"]
            batch = cfg.get("batch", 1)
            stats = {m: mean_std([r["test"][m] for r in runs])
                     for m in ("relative_l2", "mae_C", "peak_temp_error_C", "hotspot_localization_cells")}
            curves.setdefault(label, {})[n] = stats["relative_l2"]
            rows.append({
                "model": label, "sensors": n, "seeds": len(runs), "steps": runs[0]["steps"], "batch": batch,
                "chips_seen": runs[0]["steps"] * batch,
                "relative_l2_pct": f"{stats['relative_l2'][0] * 100:.2f} ± {stats['relative_l2'][1] * 100:.2f}",
                "mae_C": f"{stats['mae_C'][0]:.2f}",
                "peak_temp_error_C": f"{stats['peak_temp_error_C'][0]:.2f}",
                "hotspot_localization_cells": f"{stats['hotspot_localization_cells'][0]:.1f}",
                "parameters": runs[0]["parameters"],
            })
    if not rows:
        raise SystemExit("no results found")
    for n, value in DEEPONET_REFERENCE.items():
        rows.append({"model": "DeepONet (teammates, reference)", "sensors": n, "seeds": 5, "steps": 6400, "batch": 32,
                     "chips_seen": 200000, "relative_l2_pct": f"{value * 100:.2f}", "mae_C": "", "peak_temp_error_C": "",
                     "hotspot_localization_cells": "", "parameters": 118017})
    curves["DeepONet (teammates, reference)"] = {n: (v, 0.0) for n, v in DEEPONET_REFERENCE.items()}

    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| Model | Sensors | Seeds | Steps | Batch | Chips seen | Relative L2 (%) | MAE (°C) | Peak error (°C) | Hotspot error (cells) | Params |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['sensors']} | {r['seeds']} | {r['steps']} | {r['batch']} | {r['chips_seen']} "
                     f"| {r['relative_l2_pct']} | {r['mae_C']} | {r['peak_temp_error_C']} | {r['hotspot_localization_cells']} | {r['parameters']} |")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # The figure shows the main models only; every run is in summary.md / summary.csv.
    main = ["FNO/PINO (final)",
            "FBPINN fast (batch 32, 1024 points; 5 sensors: 10k steps, 8 and 10: 6k)",
            "XPINN fast, physics off",
            "XPINN fast, physics on, tile-edge nodes masked",
            "DeepONet (teammates, reference)"]
    colours = {m: f"C{i}" for i, m in enumerate(main)}
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, by_n in curves.items():
        if label not in main:
            continue
        xs = [n for n in SENSOR_COUNTS if n in by_n]
        ys = [by_n[n][0] * 100 for n in xs]
        style = "--" if "reference" in label else "-"
        ax.plot(xs, ys, style, marker="o", color=colours[label], label=label)
    ax.set_xlabel("number of sensors")
    ax.set_ylabel("test relative L2 error (%)")
    ax.set_xticks(SENSOR_COUNTS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Test error against sensor count (lower is better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_vs_sensors.png", dpi=150)
    print(f"wrote {len(rows)} rows and figure to {OUT_DIR}")


if __name__ == "__main__":
    main()
