"""Summarise results/real_hybrid: tables, the validation-selected hybrid, paired comparison, chart.

"The hybrid" is ONE variant for every setting: the hybrid variant with the lowest mean *validation* error over all
processors, sensor counts and seeds. Test frames play no part in that choice. (A first rule, picking the best hybrid per setting
on validation, was dropped because the three hybrids are indistinguishable on validation, so it chose almost at random; all
variants' test numbers are still in summary_hybrid.csv.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results/real_hybrid")
NAMES = {"i5": "i5-3337U", "i7": "i7-8650U"}
HYB = ["Hybrid 1: blend", "Hybrid 2: DeepONet + physics correction", "Hybrid 3: blend + physics correction"]
rng = np.random.default_rng(0)


def main() -> None:
    allr, tables, chosen_rows = [], [], []
    allruns = pd.concat([pd.read_csv(R / p / "summary_runs.csv").query("seed >= 0") for p in NAMES])
    vmean = allruns.groupby("model").val_rise_error.mean()
    GLOBAL_BEST = vmean[HYB].idxmin()
    print("mean validation error per model:", vmean.round(4).to_dict(), "-> hybrid =", GLOBAL_BEST)
    for p in NAMES:
        runs = pd.read_csv(R / p / "summary_runs.csv"); runs["processor"] = p
        fr = pd.read_csv(R / p / "per_frame_rise_error.csv")
        seeded = runs[runs.seed >= 0]
        for n, g in seeded.groupby("n_sensors"):
            m = g.groupby("model")[["rel_l2_rise", "mae_C", "rmse_C", "val_rise_error"]].agg(["mean", "std"])
            hyb = m.loc[HYB, ("val_rise_error", "mean")]
            best = GLOBAL_BEST
            for name in ["DeepONet", "Physics"] + HYB:
                tables.append({"processor": NAMES[p], "sensors": n, "model": name, "chosen_hybrid": name == best,
                               "test_rise": m.loc[name, ("rel_l2_rise", "mean")], "test_rise_std": m.loc[name, ("rel_l2_rise", "std")],
                               "test_mae_C": m.loc[name, ("mae_C", "mean")], "test_mae_std": m.loc[name, ("mae_C", "std")],
                               "val_rise": m.loc[name, ("val_rise_error", "mean")]})
            # paired comparison on test frames (seed-averaged per frame)
            f = fr[fr.n_sensors == n].groupby(["model", "frame"]).rel_l2_rise.mean().unstack(0)
            for ref in ("DeepONet", "Physics"):
                d = (f[ref] - f[best]).values
                bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                chosen_rows.append({"processor": NAMES[p], "sensors": n, "chosen_hybrid": best, "versus": ref,
                                    "frames_better": float((d > 0).mean()), "mean_gain": d.mean(), "ci_lo": lo, "ci_hi": hi,
                                    "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "tie")})
        gp = runs[(runs.seed == -1) & (runs.model == "gappy POD")].set_index("n_sensors").rel_l2_rise
        for n, v in gp.items():
            tables.append({"processor": NAMES[p], "sensors": n, "model": "gappy POD (reference)", "chosen_hybrid": False,
                           "test_rise": v, "test_rise_std": np.nan, "test_mae_C": runs[(runs.seed == -1) & (runs.model == "gappy POD") & (runs.n_sensors == n)].mae_C.iloc[0],
                           "test_mae_std": np.nan, "val_rise": np.nan})
    T = pd.DataFrame(tables); T.to_csv(R / "summary_hybrid.csv", index=False)
    C = pd.DataFrame(chosen_rows); C.to_csv(R / "paired_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False)); print(); print(C.round(4).to_string(index=False))

    # chart: DeepONet, Physics, chosen hybrid per (processor, N)
    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.axisbelow": True})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, p in zip(axes, NAMES.values()):
        t = T[T.processor == p]; w = 0.26
        for k, (lab, col) in enumerate([("DeepONet", BLUE), ("Physics", ORANGE), ("Hybrid (validation-chosen)", AQUA)]):
            vals, errs = [], []
            for n in (5, 8, 10):
                s = t[t.sensors == n]
                r = s[s.chosen_hybrid] if lab.startswith("Hybrid") else s[s.model == lab]
                vals.append(r.test_rise.iloc[0]); errs.append(r.test_rise_std.iloc[0])
            xs = np.arange(3) + (k - 1) * w
            ax.bar(xs, vals, w * 0.92, color=col, label=lab, yerr=errs, capsize=2, error_kw={"lw": 0.8})
            for x, v in zip(xs, vals):
                ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["5 sensors", "8 sensors", "10 sensors"]); ax.set_title(p, loc="left", fontsize=10)
    axes[0].set_ylabel("Relative error vs temperature rise\n(test frames, lower is better)")
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Real thermal maps: DeepONet vs physics reconstruction vs hybrid (mean of 5 seeds, whisker = std)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "results_hybrid.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
