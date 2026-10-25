"""Side-by-side map comparison on the real thermal data: truth, DeepONet, gappy POD and their errors.

For every processor it retrains the DeepONet with the settings of ``configs/real_deeponet.yaml`` (one seed, one sensor
count) and shows the test frame with the **median** DeepONet error, so the example is typical, not hand-picked.

    python -m src.eval.real_compare_maps --sensors 8 --seed 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap

from src.data.real_thermal import PROCESSORS
from src.train.train_real_deeponet import baselines, prepare, read_sensors, train_one

NAMES = {"i5": "i5-3337U", "i7": "i7-8650U", "r7_4800u": "Ryzen 7 4800U", "rtx4060": "RTX 4060", "tpu": "Coral TPU"}
WARM = LinearSegmentedColormap.from_list("warm", ["#fde9de", "#f7b595", "#eb6834", "#b8431b", "#6e2208"])
COOL = LinearSegmentedColormap.from_list("cool", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--sensors", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processors", default=",".join(PROCESSORS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    procs = a.processors.split(",")
    out = Path(a.out or f"results/real_deeponet/comparison_maps_n{a.sensors}.png")
    n = a.sensors
    fig = plt.figure(figsize=(12.5, 2.55 * len(procs) + 0.7))
    gs = fig.add_gridspec(len(procs), 5, wspace=0.55, hspace=0.14, top=0.955, bottom=0.01, left=0.07, right=0.95)
    for r, proc in enumerate(procs):
        D = prepare(proc, cfg)
        readings = {k: read_sensors(D["maps"][k], D["rc"], D["frames"][k][0][1] if k != "train" and len({w for w, _ in D["frames"][k]}) == 1 else 0,
                                    n, cfg, sid) for sid, k in enumerate(("train", "validation", "test"))}
        _, gp = baselines(D, n, cfg, readings, return_preds=True)
        rows, best, secs, _, maps = train_one(D, n, a.seed, cfg, readings, quiet=True, save_maps=len(D["maps"]["test"]))
        pred = np.stack([m[1] for m in maps]); true = D["maps"]["test"]
        mask = np.ones(true.shape[1:], bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
        rmse = np.array([np.sqrt(np.mean((pred[k] - true[k])[mask] ** 2)) for k in range(len(true))])
        k = int(np.argsort(rmse)[len(rmse) // 2])
        t, d, g = true[k], pred[k], gp[k]
        ed, eg = np.abs(d - t), np.abs(g - t)
        vmin, vmax, emax = t.min(), t.max(), max(ed.max(), eg.max())
        stats = lambda e: f"MAE {e[mask].mean():.2f} C"
        panels = [(t, "True map", WARM, (vmin, vmax)), (d, f"DeepONet  ({stats(ed)})", WARM, (vmin, vmax)),
                  (g, f"Gappy POD  ({stats(eg)})", WARM, (vmin, vmax)), (ed, "DeepONet |error|", COOL, (0, emax)),
                  (eg, "Gappy POD |error|", COOL, (0, emax))]
        for c, (img, title, cm, lim) in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            im = ax.imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1])
            ax.scatter(D["rc"][:n, 1], D["rc"][:n, 0], s=14, c="white", edgecolors="#0b0b0b", linewidths=0.6)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=8)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(f"{NAMES[proc]}\ntest frame {k}", fontsize=9)
            if c in (2, 4):
                cb = fig.colorbar(im, cax=ax.inset_axes([1.05, 0, 0.05, 1])); cb.ax.tick_params(labelsize=6); cb.set_label("deg C", fontsize=6)
        print(f"{proc}: frame {k} of {len(true)} (median error), DeepONet MAE {ed[mask].mean():.2f} C, gappy POD MAE {eg[mask].mean():.2f} C", flush=True)
    fig.suptitle(f"Real thermal maps, {n} sensors (white dots): ground truth vs DeepONet vs gappy POD, one typical test frame per processor "
                 f"(seed {a.seed})", x=0.01, y=0.995, ha="left", fontsize=10)
    fig.text(0.01, 0.972, "The three left panels share one temperature scale (color bar on the gappy POD panel); each error pair shares one scale.", fontsize=8, color="#52514e")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
