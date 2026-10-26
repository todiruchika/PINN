"""Table, error-vs-sensors figure and heatmaps for the real IR-camera experiments.

Reads results/real/<model>/N<n>_seed0/results.json and results/real/baselines.json, writes to results/real/:
    summary.md, error_vs_sensors.png, heatmaps_N<n>.png
Usage: python -m src.eval.real_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from src.data.real_thermal import GRID, RealThermal
from src.models.fbpinn import FBPINN, grid_coordinates
from src.models.phys_decoder import SensorToField
from src.sensors.readers import sensor_mask
from src.train import fbpinn as fbpinn_trainer
from src.train import phys_hybrid

OUT = Path("results/real")
MODELS = {"decoder": "Data-only decoder", "fbpinn": "FBPINN", "xpinn": "XPINN", "fno": "FNO/PINO"}
BASELINES = {"tps": "Thin-plate spline (no training)", "pod_fit": "POD fit (no network)", "mean_map": "Mean training map"}
SENSORS = (5, 8, 10)


def load(model: str, n: int) -> dict | None:
    path = OUT / model / f"N{n}_seed0" / "results.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table() -> dict[str, dict[int, dict]]:
    base = json.loads((OUT / "baselines.json").read_text(encoding="utf-8"))
    rows: dict[str, dict[int, dict]] = {}
    for key, label in MODELS.items():
        rows[label] = {n: r["test"] for n in SENSORS if (r := load(key, n))}
    for key, label in BASELINES.items():
        rows[label] = {n: base[f"N{n}"]["test"][key] for n in SENSORS}
    lines = ["| Method | Sensors | Rel. L2 (%) | MAE (°C) | RMSE (°C) | Peak error (°C) | Hotspot error (cells) |", "|---|---|---|---|---|---|---|"]
    for label, by_n in rows.items():
        for n, m in by_n.items():
            lines.append(f"| {label} | {n} | {m['relative_l2'] * 100:.2f} | {m['mae_C']:.2f} | {m['rmse_C']:.2f} | "
                         f"{m['peak_temp_error_C']:.2f} | {m['hotspot_localization_cells']:.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def error_figure(rows: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in ((axes[0], "relative_l2", "Test relative L2 error (%)"), (axes[1], "mae_C", "Test MAE (°C)")):
        for i, (label, by_n) in enumerate(rows.items()):
            if label == BASELINES["mean_map"]:
                continue
            xs = [n for n in SENSORS if n in by_n]
            scale = 100 if metric == "relative_l2" else 1
            ax.plot(xs, [by_n[n][metric] * scale for n in xs], "--" if label in BASELINES.values() else "-", marker="o", label=label)
        ax.set_xticks(SENSORS)
        ax.set_xlabel("number of sensors")
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Real IR-camera data (i7-8650U), held-out workloads. Lower is better; dashed = no neural network", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "error_vs_sensors.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def predictions(real: RealThermal, data: dict, n: int) -> dict[str, torch.Tensor]:
    out = {}
    fb = load("fbpinn", n)
    if fb:
        model = FBPINN(n_sensors=n)
        model.load_state_dict(torch.load(OUT / "fbpinn" / f"N{n}_seed0" / "best.pt"))
        out["FBPINN"] = fbpinn_trainer.predict_fields(model, data, grid_coordinates())
    if load("decoder", n):
        model = SensorToField(n, "ml")
        model.load_state_dict(torch.load(OUT / "decoder" / f"N{n}_seed0" / "best.pt"))
        out["Data-only decoder"] = phys_hybrid.predict_fields(model, data)
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    pts = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
    tps = []
    for i in range(len(data["ids"])):
        pos = data["positions"][i].numpy() / 0.012 * (GRID - 1)
        tps.append(torch.tensor(RBFInterpolator(pos[:, ::-1], data["readings"][i].numpy(), kernel="thin_plate_spline",
                                                smoothing=1.0)(pts).reshape(GRID, GRID), dtype=torch.float32))
    out["Thin-plate spline"] = torch.stack(tps)
    return out


def heatmaps(real: RealThermal, n: int) -> None:
    test_ids = real.splits["test"]
    chosen = [test_ids[i] for i in (50, 150, 250)]       # one mid-recording frame from each test workload
    data = real.arrays(chosen, n)
    preds = predictions(real, data, n)
    cols = ["Ground truth (IR camera)"] + list(preds) + [f"|error| {k}" for k in preds]
    fig, axes = plt.subplots(len(chosen), len(cols), figsize=(2.6 * len(cols), 2.6 * len(chosen)))
    for r in range(len(chosen)):
        truth = data["T"][r].numpy()
        lo, hi = truth.min(), truth.max()
        mask = sensor_mask(data["positions"][r].tolist())
        ys, xs = np.where(mask)
        images = [truth] + [p[r].numpy() for p in preds.values()]
        errs = [np.abs(p[r].numpy() - truth) for p in preds.values()]
        for c, img in enumerate(images + errs):
            ax = axes[r, c]
            is_err = c >= len(images)
            im = ax.imshow(img, cmap="viridis" if is_err else "inferno", vmin=0 if is_err else lo,
                           vmax=max(e.max() for e in errs) if is_err else hi)
            if c == 0:
                ax.scatter(xs, ys, s=12, c="cyan", marker="x")
                ax.set_ylabel(chosen[r][1], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
        fig.colorbar(im, ax=axes[r, -1], fraction=0.046)
    fig.suptitle(f"{n} sensors (x), held-out workloads; temperature maps in °C (error panels share one scale per row)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"heatmaps_N{n}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    rows = table()
    error_figure(rows)
    real = RealThermal()
    for n in SENSORS:
        heatmaps(real, n)
    print("wrote summary.md, error_vs_sensors.png, heatmaps_N5/8/10.png")


if __name__ == "__main__":
    main()
