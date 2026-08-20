"""Render the Model B design-flow diagram to results/figures/design_flow.png.

Presentation/report asset. Regenerate with:
    python src/make_flowchart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import resolve_path  # noqa: E402

# palette: inputs / model / training / evaluation (ground truth) / outputs
C_IN, E_IN = "#dbe9f6", "#2f6fa8"
C_MODEL, E_MODEL = "#dff0d8", "#3d7a35"
C_TRAIN, E_TRAIN = "#fdebd0", "#b9770e"
C_EVAL, E_EVAL = "#f9dcdc", "#a93226"
C_OUT, E_OUT = "#e8e8e8", "#555555"


def box(ax, x, y, w, h, title, lines=(), face=C_IN, edge=E_IN, file=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                               facecolor=face, edgecolor=edge, linewidth=1.6, zorder=2))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", zorder=3,
            fontsize=10.5, fontweight="bold", color=edge)
    for i, line in enumerate(lines):
        ax.text(x + 0.24, y + h - 0.72 - i * 0.345, line, ha="left", va="top",
                fontsize=8.7, zorder=3, color="#222222")
    if file:
        ax.text(x + w - 0.18, y + 0.14, file, ha="right", va="bottom", zorder=3,
                fontsize=7.4, style="italic", family="monospace", color=edge)
    return (x + w / 2, y), (x + w / 2, y + h)


def arrow(ax, p_from, p_to, color="#333333"):
    ax.add_patch(FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=16,
                                linewidth=1.5, color=color, zorder=1, shrinkA=2, shrinkB=2))


fig = plt.figure(figsize=(17.5, 12.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.04, 1.0], wspace=0.05,
                      left=0.015, right=0.985, top=0.918, bottom=0.03)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24.6)
    ax.axis("off")

fig.suptitle("Model B  —  Physics-Only PINN:  Design Flow", fontsize=18, fontweight="bold", y=0.977)
fig.text(0.5, 0.938,
         "one independent solve per (chip x sensor count x seed)  —  no training set, no cross-chip learning",
         ha="center", fontsize=11, style="italic", color="#444444")

# ----------------------------- LEFT: main pipeline ------------------------- #
axL.text(5, 24.2, "A.   End-to-end pipeline", ha="center", fontsize=13.5,
         fontweight="bold", color="#111111")
W, X = 8.7, 0.65

_, t1 = box(axL, X, 21.4, W, 2.35, "1.   INPUTS  —  everything the model may see", [
    "5 / 8 / 10 sensor readings  (x, y, T)          <-  sensors/*.json",
    "k = 142.5 W/(m K),  T_boundary = 27.3 degC     <-  metadata.csv",
    "the governing equation   grad^2 T = -Q / k     (needs no data)",
], C_IN, E_IN, "data_loader.py : load_chip()")

b2, t2 = box(axL, X, 19.05, W, 1.8, "2.   NON-DIMENSIONALISE", [
    "T_norm = (T - T_boundary) / scale       f_norm = Q / (k * scale)",
    "scale = hottest sensor - T_boundary   <-  from SENSORS ONLY",
], C_IN, E_IN, "data_loader.py : ChipScaler")

b3, t3 = box(axL, X, 16.05, W, 2.45, "3.   BUILD THE TWO NETWORKS", [
    "FIELD    (x,y) -> T_norm     MLP 4 x 64 tanh      12,737 knobs",
    "         hard BC:   T_norm = d(x,y) * NN(x,y)",
    "                    d = 16 x(1-x) y(1-y)  = 0 on all 4 edges",
    "SOURCE   6 Gaussian blobs (bright, x, y, width)       24 knobs",
], C_MODEL, E_MODEL, "models/pinn.py : build_pinn()")

b4, t4 = box(axL, X, 13.45, W, 2.05, "4.   ADAM  —  5,000 steps", [
    "2,048 fresh random collocation points every step",
    "lr 1e-3, cosine decay to 1e-5            coarse shaping",
], C_TRAIN, E_TRAIN, "train_pinn.py : train_one_chip()")

b5, t5 = box(axL, X, 10.85, W, 2.05, "5.   L-BFGS  —  2,000 steps", [
    "collocation set FROZEN at 4,096 points",
    "(resampling would break its line search)    fine polish",
], C_TRAIN, E_TRAIN, "train_pinn.py")

b6, t6 = box(axL, X, 9.15, W, 1.35, "6.   RESTORE BEST ITERATE", [
    "L-BFGS can step uphill — keep the best surface ever seen",
], C_TRAIN, E_TRAIN, "restore_best")

b7, t7 = box(axL, X, 6.95, W, 1.7, "7.   PREDICT THE FULL FIELD", [
    "evaluate the trained field at all 128 x 128 grid points, then",
    "invert the scaling:   T = T_boundary + scale * T_norm",
], C_OUT, E_OUT, "train_pinn.py : predict_field()")

b8, t8 = box(axL, X, 4.75, W, 1.75, "8.   ===  GROUND TRUTH LOADED HERE  ===", [
    "matrices/<chip>.npy  —  the 128 x 128 finite-element answer",
    "first and only time it is touched:  scoring, never training",
], C_EVAL, E_EVAL, "data_loader.py : chip.field()")

b9, t9 = box(axL, X, 2.5, W, 1.8, "9.   SCORE", [
    "rel L2   |   max abs err   |   peak-location err   |   boundary MAE",
    "sensor RMSE   |   inferred-source err   |   wall-clock time",
], C_EVAL, E_EVAL, "evaluate.py : field_metrics()")

b10, t10 = box(axL, X, 0.3, W, 1.75, "10.   ARTEFACTS", [
    "results/metrics.csv (one row)    results/runs/*.json (loss history)",
    "results/figures/*.png    (true | predicted | error  triptych)",
], C_OUT, E_OUT, "evaluate.py / save_triptych()")

for below, above in [(b2, t1), (b3, t2), (b4, t3), (b5, t4), (b6, t5),
                     (b7, t6), (b8, t7), (b9, t8), (b10, t9)]:
    arrow(axL, above, below)

axL.add_patch(FancyArrowPatch((X + W + 0.16, 14.4), (X + W + 0.16, 11.8), arrowstyle="-",
                             linewidth=1.2, color=E_TRAIN, linestyle=(0, (4, 3))))
axL.text(X + W + 0.28, 13.1, "7,000 steps\ntotal\n~14 min\nper chip", fontsize=8.2,
         color=E_TRAIN, va="center", style="italic")

# ----------------------------- RIGHT: inner step -------------------------- #
axR.text(5, 24.2, "B.   Inside ONE training step   (runs 7,000x per chip)",
         ha="center", fontsize=13.5, fontweight="bold", color="#111111")
W2, X2 = 8.35, 0.5

_, s1 = box(axR, X2, 22.15, W2, 1.4, "(i)   SAMPLE", [
    "throw 2,048 random points into the unit square",
], C_TRAIN, E_TRAIN, "losses.py : sample_interior()")

box(axR, X2, 17.6, 4.0, 4.05, "(ii)   PHYSICS BADNESS", [
    "field net  ->  T_norm at the",
    "2,048 points",
    "autodiff ONCE   -> dT/dx, dT/dy",
    "autodiff TWICE  -> grad^2 T_norm",
    "source model    -> f_norm",
    "",
    "residual = grad^2 T_norm + f_norm",
    "L_res = mean( residual^2 )",
], C_MODEL, E_MODEL, "losses.py")

box(axR, X2 + 4.85, 17.6, 4.0, 4.05, "(iii)   DATA BADNESS", [
    "ask the field for T at the",
    "5 sensor positions",
    "",
    "compare with the 5 measured",
    "temperatures",
    "",
    "L_data = mean( (pred - meas)^2 )",
], C_IN, E_IN, "losses.py : data_loss()")

arrow(axR, s1, (X2 + 2.0, 21.65))
arrow(axR, s1, (X2 + 6.85, 21.65))

_, s4 = box(axR, X2, 15.3, W2, 1.6, "(iv)   TOTAL BADNESS", [
    "L  =  1.0 x L_res   +   100 x L_data",
    "the 100 stops 5 data points being drowned by 2,048 physics points",
], C_TRAIN, E_TRAIN, "w_data in the config")
arrow(axR, (X2 + 2.0, 17.6), (X2 + 3.1, 16.9))
arrow(axR, (X2 + 6.85, 17.6), (X2 + 5.7, 16.9))

_, s5 = box(axR, X2, 13.3, W2, 1.45, "(v)   BACKPROPAGATE", [
    "for each of 12,761 knobs:  does nudging you up make L worse?",
], C_TRAIN, E_TRAIN, "loss.backward()")
arrow(axR, (X2 + W2 / 2, 15.3), (X2 + W2 / 2, 14.75))

_, s6 = box(axR, X2, 11.3, W2, 1.45, "(vi)   UPDATE", [
    "nudge every knob the improving way;   shrink the step size",
], C_TRAIN, E_TRAIN, "optimizer.step() / scheduler.step()")
arrow(axR, (X2 + W2 / 2, 13.3), (X2 + W2 / 2, 12.75))

axR.add_patch(FancyArrowPatch((X2 + W2 + 0.20, 12.02), (X2 + W2 + 0.20, 22.85),
                             arrowstyle="-|>", mutation_scale=15, linewidth=1.6,
                             color=E_TRAIN, connectionstyle="arc3,rad=-0.10"))
axR.text(X2 + W2 + 0.52, 17.4, "repeat 7,000x", fontsize=8.8, color=E_TRAIN,
         style="italic", rotation=90, ha="center", va="center")

box(axR, X2, 5.95, W2, 4.75, "WHY THIS NEEDS NO ANSWER KEY", [
    "The heat equation is a RULE, checkable at any point without a",
    "reference solution — like spotting two 7s in one Sudoku row.",
    "The curvature comes from differentiating the network's own",
    "formula (autodiff), not from comparing against stored data.",
    "",
    "Physics alone:   infinitely many valid fields (one per source).",
    "5 sensors alone: infinitely many surfaces through 5 points.",
    "TOGETHER + the exact boundary:  the answer is pinned down.",
    "",
    "Consequence:  the field comes out well (~13% error) but the",
    "source shape does not (~85%) — many layouts fit 5 readings.",
], "#eef3fa", "#2f4f6f")

box(axR, X2, 0.3, W2, 5.35, "WHERE THE HARD CONSTRAINT LIVES", [
    "NOT a step in the loop and NOT a loss term. It is baked into",
    "the function being optimised:",
    "",
    "        T  =  T_boundary  +  d(x,y) * NN(x,y)",
    "        d  =  16 x(1-x) y(1-y)    ->    d = 0 on every edge",
    "",
    "At any boundary point the network is multiplied by zero:",
    "        T = T_boundary + 0 * (anything) = T_boundary  exactly.",
    "",
    "The network only predicts the RISE ABOVE AMBIENT, and that rise",
    "is forced to vanish at the edges. A non-zero boundary is fine —",
    "it only has to be KNOWN, not zero.",
    "Result: boundary_mae = 0.0 on all 18 runs, no weight to tune.",
], "#eaf5ea", E_MODEL)

out = resolve_path("results/figures/design_flow.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170, facecolor="white")
print(f"wrote {out}")
