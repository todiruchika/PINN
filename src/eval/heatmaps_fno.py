"""Ground-truth vs generated heatmaps for the FNO/PINO model, with error maps.

Usage (from the repo root):
    python -m src.eval.heatmaps_fno results/fno_pino_wsrc0.1/N10_seed0/best.pt <out.png> 4
"""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.splits import load_splits
from src.data.dataset import load_metadata
from src.models.fno_pino import FNOPINO
from src.train.fno_pino import load_split_arrays, predict_fields
from src.sensors.readers import metre_to_idx
from src.eval.metrics import relative_l2
from src.sensors.readers import sensor_mask

ckpt, out_png, n_chips = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
splits = load_splits(); meta = load_metadata()
test = load_split_arrays(splits["test"][:n_chips], 10, meta)
model = FNOPINO(width=32, modes=8, depth=4)
model.load_state_dict(torch.load(ckpt))
pred = predict_fields(model, test).numpy()
fig, axes = plt.subplots(n_chips, 3, figsize=(10, 3.4 * n_chips))
for i in range(n_chips):
    true = test["T"][i].numpy(); p = pred[i]
    mask = ~sensor_mask(test["positions"][i].tolist())
    err = np.where(mask, p - true, np.nan)
    vmin, vmax = true.min(), true.max()
    sensors = [(metre_to_idx(x), metre_to_idx(y)) for x, y in test["positions"][i].tolist()]
    titles = [f"ground truth (peak {true.max():.1f} C)", f"generated (peak {p.max():.1f} C)",
              f"error (rel L2 {100*relative_l2(p, true, mask):.1f}%)"]
    for j, ax in enumerate(axes[i]):
        if j < 2:
            im = ax.imshow((true, p)[j], vmin=vmin, vmax=vmax, cmap="inferno")
        else:
            lim = np.nanmax(np.abs(err))
            im2 = ax.imshow(err, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.scatter([c for c, r in sensors], [r for c, r in sensors], s=12, facecolors="none", edgecolors="cyan")
        ax.set_title(titles[j], fontsize=9); ax.axis("off")
    fig.colorbar(im, ax=axes[i, :2].tolist(), fraction=0.03, label="C")
    fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, label="C")
fig.suptitle("FNO/PINO, 10 sensors (cyan circles), test chips", fontsize=11)
fig.tight_layout(); fig.savefig(out_png, dpi=110)
print("saved", out_png)
