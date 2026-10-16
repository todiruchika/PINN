"""Ground-truth vs generated heatmaps for the FBPINN / XPINN models, with error maps.

Same layout as src/eval/heatmaps_fno.py, so the two models' figures can be compared side by side.

Usage (from the repo root):
    python -m src.eval.heatmaps_fbpinn results/fbpinn/N5_seed0/best.pt <out.png> 4
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset import load_metadata
from src.data.splits import load_splits
from src.eval.metrics import relative_l2
from src.models.fbpinn import FBPINN, grid_coordinates
from src.sensors.readers import metre_to_idx, sensor_mask
from src.train.fbpinn import predict_fields
from src.models.pod_prior import fit_basis
from src.models.xpinn import XPINN
from src.train.fno_pino import attach_prior, load_split_arrays


def main() -> None:
    ckpt, out_png, n_chips = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    run_config = json.loads((ckpt.parent / "results.json").read_text(encoding="utf-8"))["config"]
    n_sensors = int(run_config["sensors"])
    splits = load_splits()
    meta = load_metadata()
    test = load_split_arrays(splits["test"][:n_chips], n_sensors, meta)
    if run_config.get("prior", "none") == "pod":
        # Same training-only basis the run used: refit it on the training chips.
        train = load_split_arrays(splits["train"], n_sensors, meta)
        mean, basis = fit_basis(train["T"].numpy(), train["T_amb"].numpy(), int(run_config["prior_rank"]))
        attach_prior(test, mean, basis)
    kind = run_config.get("model", "fbpinn")
    if kind == "xpinn":
        model = XPINN(n_sensors=n_sensors, tiles_per_side=run_config["windows"], embed_dim=run_config["embed"],
                      hidden=run_config["hidden"], interface_points=run_config.get("interface_points", 16))
    else:
        model = FBPINN(n_sensors=n_sensors, windows_per_side=run_config["windows"],
                       window_width=run_config["window_width"], embed_dim=run_config["embed"],
                       hidden=run_config["hidden"])
    model.load_state_dict(torch.load(ckpt))
    pred = predict_fields(model, test, grid_coordinates()).numpy()

    fig, axes = plt.subplots(n_chips, 3, figsize=(10, 3.4 * n_chips))
    for i in range(n_chips):
        true = test["T"][i].numpy()
        p = pred[i]
        mask = ~sensor_mask(test["positions"][i].tolist())
        err = np.where(mask, p - true, np.nan)
        vmin, vmax = true.min(), true.max()
        sensors = [(metre_to_idx(x), metre_to_idx(y)) for x, y in test["positions"][i].tolist()]
        titles = [f"ground truth (peak {true.max():.1f} C)", f"generated (peak {p.max():.1f} C)",
                  f"error (rel L2 {100 * relative_l2(p, true, mask):.1f}%)"]
        for j, ax in enumerate(axes[i]):
            if j < 2:
                im = ax.imshow((true, p)[j], vmin=vmin, vmax=vmax, cmap="inferno")
            else:
                lim = np.nanmax(np.abs(err))
                im2 = ax.imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim)
            ax.scatter([c for c, r in sensors], [r for c, r in sensors], s=12, facecolors="none", edgecolors="cyan")
            ax.set_title(titles[j], fontsize=9)
            ax.axis("off")
        fig.colorbar(im, ax=axes[i, :2].tolist(), fraction=0.03, label="C")
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, label="C")
    fig.suptitle(f"{kind.upper()}, {n_sensors} sensors (cyan circles), test chips", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("saved", out_png)


if __name__ == "__main__":
    main()
