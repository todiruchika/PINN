"""Build the paper's figures.

Numbers are copied from the result files on the project branches (sources in README.md):
  - low-data study: results/low_data/summary.md on branch smriti-hybrid(pinn+deeponet)
Run from the paper root:  python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

FIG = Path("figures")

# Test relative L2 (%), 100 test chips. DeepONet = mean of 5 seeds; PINN does not use training chips; blend uses DeepONet seed 0.
CHIPS = [25, 50, 100, 500]
LOW_DATA = {
    5: {"DeepONet": [13.83, 13.34, 12.44, 11.46], "PINN": [13.33] * 4, "Hybrid (blend)": [12.76, 12.51, 12.21, 11.62]},
    8: {"DeepONet": [12.74, 11.34, 9.52, 8.62], "PINN": [8.80] * 4, "Hybrid (blend)": [8.78, 8.72, 8.53, 8.21]},
    10: {"DeepONet": [9.33, 7.26, 6.42, 6.13], "PINN": [6.51] * 4, "Hybrid (blend)": [6.44, 6.23, 6.11, 6.01]},
}
STYLE = {  # validated categorical slots 1-3, plus marker and dash so identity never rests on colour alone (print, CVD)
    "DeepONet": dict(color="#2a78d6", marker="o", linestyle="-"),
    "PINN": dict(color="#eb6834", marker="s", linestyle="--"),
    "Hybrid (blend)": dict(color="#1baf7a", marker="^", linestyle="-"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def low_data_figure() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), constrained_layout=True)
    for ax, (n, series) in zip(axes, LOW_DATA.items()):
        for name, ys in series.items():
            ax.plot(CHIPS, ys, label=name, linewidth=1.4, markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, **STYLE[name])
        ax.set_xscale("log")
        ax.set_xticks(CHIPS, [str(c) for c in CHIPS])
        ax.minorticks_off()
        ax.set_title(f"{n} sensors", color=INK, fontsize=8.5)
        ax.set_xlabel("Training chips")
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Test relative $L_2$ error (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=3, fontsize=7.5)
    fig.savefig(FIG / "low_data.pdf")
    fig.savefig(FIG / "low_data.png", dpi=200)


# Real IR maps, i7, sampled test frames: error against temperature rise vs training workloads the DeepONet sees.
# Source: results/real_fbpinn/summary.csv and README on branch siya-real-fair (PINN/FBPINN use no training maps).
WORKLOADS = [2, 4, 13]
REAL_LOW = {
    5: {"DeepONet": [0.504, 0.439, 0.230], "PINN": [0.695] * 3, "FBPINN": [0.608] * 3,
        "Hybrid (correction)": [0.444, 0.448, 0.229]},
    8: {"DeepONet": [0.432, 0.356, 0.110], "PINN": [0.371] * 3, "FBPINN": [0.398] * 3,
        "Hybrid (correction)": [0.196, 0.163, 0.106]},
    10: {"DeepONet": [0.403, 0.338, 0.107], "PINN": [0.176] * 3, "FBPINN": [0.168] * 3,
         "Hybrid (correction)": [0.191, 0.143, 0.091]},
}
REAL_STYLE = {
    "DeepONet": STYLE["DeepONet"],
    "PINN": STYLE["PINN"],
    "FBPINN": dict(color="#eda100", marker="D", linestyle=":"),
    "Hybrid (correction)": STYLE["Hybrid (blend)"],
}


def real_low_data_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), constrained_layout=True)
    for ax, (n, series) in zip(axes, REAL_LOW.items()):
        for name, ys in series.items():
            ax.plot(WORKLOADS, ys, label=name, linewidth=1.4, markersize=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, **REAL_STYLE[name])
        ax.set_xscale("log")
        ax.set_xticks(WORKLOADS, [str(w) for w in WORKLOADS])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(f"{n} sensors", color=INK, fontsize=8.5)
        ax.set_xlabel("Training workloads (DeepONet)")
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Test error vs. temperature rise")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="outside upper center", ncol=4, fontsize=7.5)
    fig.savefig(FIG / "real_low_data.pdf")
    fig.savefig(FIG / "real_low_data.png", dpi=200)


def crop(name: str, top_frac: float, out: str) -> None:
    """Remove the title strip a source figure was saved with; captions go in the paper instead."""
    im = Image.open(FIG / name)
    w, h = im.size
    im.crop((0, int(h * top_frac), w, h)).save(FIG / out)


if __name__ == "__main__":
    low_data_figure()
    real_low_data_figure()
    crop("real_maps_n8.png", 0.11, "real_maps_n8_crop.png")
    crop("sim_maps_n10.png", 0.04, "sim_maps_n10_crop.png")
