"""How well do the TRUE maps satisfy the grid version of the heat equation?

Computes the finite-volume residual (src/physics/grid_residual.py) of the ground-truth temperature and
source maps for validation chips. If the grid equation matched the dataset, the residual would be near
zero. It is not, which limits what a physics loss on this grid can add.

Usage (from the repo root):
    python -m src.eval.physics_consistency
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, sobel

from generator.pde_solver import H_EFF_DEFAULT
from src.data.dataset import load_metadata, load_sample
from src.data.splits import load_splits
from src.physics.grid_residual import TEMP_SCALE_C, grid_residual

CHIPS = 40


def main() -> None:
    meta = load_metadata()
    ids = load_splits()["validation"][:CHIPS]
    raw, concentration, away_from_edges, pooled = [], [], [], {4: [], 8: [], 16: []}
    for sid in ids:
        sample = load_sample(sid, 10, meta=meta)
        ambient = float(sample["meta"]["T_amb_C"])
        theta = torch.tensor((sample["T"] - ambient) / TEMP_SCALE_C, dtype=torch.float32)[None]
        source = sample["Q"] / (H_EFF_DEFAULT * TEMP_SCALE_C)
        residual = grid_residual(theta, torch.tensor(source, dtype=torch.float32)[None], torch.tensor([ambient]))
        signal = float(torch.sqrt((theta ** 2).mean()))
        r = residual[0].numpy()
        raw.append(np.sqrt((r ** 2).mean()) / signal)
        ordered = np.sort((r ** 2).ravel())[::-1]
        concentration.append(ordered[: len(ordered) // 20].sum() / ordered.sum())
        gradient = np.hypot(sobel(source, 0), sobel(source, 1))
        edges = binary_dilation(gradient > 1e-3 * max(gradient.max(), 1e-9), iterations=4)
        away_from_edges.append(np.sqrt((r[~edges] ** 2).mean()) / signal)
        for patch in pooled:
            pooled_signal = float(torch.sqrt((F.avg_pool2d(theta[:, None], 8) ** 2).mean()))
            pooled[patch].append(float(torch.sqrt((F.avg_pool2d(residual[:, None], patch) ** 2).mean())) / pooled_signal)
    print(f"{CHIPS} validation chips, true maps through the grid heat equation")
    print(f"pointwise residual / signal:                     {100 * np.mean(raw):.1f}%")
    print(f"share of squared residual in worst 5% of pixels: {100 * np.mean(concentration):.0f}%")
    print(f"residual / signal away from source-block edges:  {100 * np.mean(away_from_edges):.1f}%")
    for patch, values in pooled.items():
        print(f"residual averaged over {patch}x{patch} patches / signal: {100 * np.mean(values):.1f}%")


if __name__ == "__main__":
    main()
