"""Simple baselines on the real IR-camera data: no neural network, nothing learned beyond a linear basis.

  mean-map     the training-frame average temperature map (ignores the sensors)
  POD fit      least-squares fit of the sensor readings to a POD basis learned on the training frames
  TPS          thin-plate-spline interpolation of the sensor readings (no training data at all)

Scores are on non-sensor nodes with the same metrics as every other model. Output: results/real/baselines.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

from src.data.real_thermal import AMBIENT_C, GRID, RealThermal
from src.models.pod_prior import fit_basis, prior_theta
from src.physics.grid_residual import TEMP_SCALE_C
from src.train.fno_pino import score


def main() -> None:
    real = RealThermal()
    out = {}
    train = real.arrays(real.splits["train"], 10)
    mean_map = train["T"].mean(0)
    pod_mean, pod_basis = fit_basis(train["T"].numpy(), train["T_amb"].numpy(), 10)
    yy, xx = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    grid_pts = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(float)
    for n in (5, 8, 10):
        test = real.arrays(real.splits["test"], n)
        val = real.arrays(real.splits["validation"], n)
        row = {}
        for split_name, data in (("validation", val), ("test", test)):
            preds = {"mean_map": mean_map.expand(len(data["ids"]), -1, -1)}
            theta = prior_theta(data["positions"].numpy(), data["readings"].numpy(), data["T_amb"].numpy(), pod_mean, pod_basis)
            preds["pod_fit"] = AMBIENT_C + TEMP_SCALE_C * theta
            tps = []
            for i in range(len(data["ids"])):
                pos = data["positions"][i].numpy() / 0.012 * (GRID - 1)            # (x, y) in grid units
                interp = RBFInterpolator(pos[:, ::-1], data["readings"][i].numpy(), kernel="thin_plate_spline", smoothing=1.0)
                tps.append(torch.tensor(interp(grid_pts).reshape(GRID, GRID), dtype=torch.float32))
            preds["tps"] = torch.stack(tps)
            row[split_name] = {k: score(v, data) for k, v in preds.items()}
        out[f"N{n}"] = row
        print(n, {k: round(v["relative_l2"] * 100, 2) for k, v in row["test"].items()},
              {k: round(v["mae_C"], 2) for k, v in row["test"].items()}, flush=True)
    Path("results/real").mkdir(parents=True, exist_ok=True)
    Path("results/real/baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
