"""Summarise results/real_pinn_hybrid: DeepONet vs Ruchika's PINN vs hybrids, full data and few training workloads (i7).

In each experiment "the hybrid" is the one hybrid variant (A blend, B correction, C blend + correction) with the lowest mean *validation*
error over that experiment's settings and seeds; test frames play no part in that choice.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results/real_pinn_hybrid")
HYB = ["Hybrid A: blend (DeepONet, PINN)", "Hybrid B: DeepONet + physics correction", "Hybrid C: blend + physics correction"]
rng = np.random.default_rng(0)


def load(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sfx = "" if exp == "full" else f"_{exp}"
    return pd.read_csv(R / f"summary_runs{sfx}.csv"), pd.read_csv(R / f"per_frame_rise_error{sfx}.csv")


def main() -> None:
    exps = ["full"] + [f"low{k}" for k in (4, 2, 1) if (R / f"summary_runs_low{k}.csv").exists()]
    rows, pairs, chosen = [], [], {}
    for exp in exps:
        runs, fr = load(exp)
        if exp != "full":
            runs, fr = runs[runs.processor == "i7"], fr[fr.processor == "i7"]
        vm = runs.groupby("model").val_rise_error.mean()
        best = vm[HYB].idxmin(); chosen[exp] = best
        print(f"[{exp}] mean validation error:", vm.round(3).to_dict(), "-> hybrid =", best)
        for (proc, n), g in runs.groupby(["processor", "n_sensors"]):
            m = g.groupby("model")[["rel_l2_rise", "mae_C", "val_rise_error"]].agg(["mean", "std"])
            for name in ["DeepONet", "PINN"] + HYB:
                rows.append({"experiment": exp, "processor": proc, "sensors": n, "model": name, "chosen_hybrid": name == best,
                             "test_rise": m.loc[name, ("rel_l2_rise", "mean")], "test_rise_std": m.loc[name, ("rel_l2_rise", "std")],
                             "test_mae_C": m.loc[name, ("mae_C", "mean")], "val_rise": m.loc[name, ("val_rise_error", "mean")]})
            f = fr[(fr.processor == proc) & (fr.n_sensors == n)].groupby(["model", "frame"]).rel_l2_rise.mean().unstack(0)
            for ref in ("DeepONet", "PINN"):
                d = (f[ref] - f[best]).values
                bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                pairs.append({"experiment": exp, "processor": proc, "sensors": n, "hybrid": best, "versus": ref, "test_frames": len(d),
                              "mean_gain": d.mean(), "ci_lo": lo, "ci_hi": hi, "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "tie")})
    T, C = pd.DataFrame(rows), pd.DataFrame(pairs)
    T.to_csv(R / "summary.csv", index=False); C.to_csv(R / "paired_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[T.chosen_hybrid | T.model.isin(["DeepONet", "PINN"])].round(3).to_string(index=False)); print(); print(C.round(4).to_string(index=False))

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.axisbelow": True})
    pick = lambda t, lab: t[t.chosen_hybrid] if lab.startswith("Hybrid") else t[t.model == lab]
    # chart 1: full data
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, proc in zip(axes, ["i5", "i7"]):
        t = T[(T.experiment == "full") & (T.processor == proc)]; w = 0.26
        for k, (lab, col) in enumerate([("DeepONet", BLUE), ("PINN", ORANGE), ("Hybrid (validation-chosen)", AQUA)]):
            v = [pick(t[t.sensors == n], lab).test_rise.iloc[0] for n in (5, 8, 10)]
            e = [pick(t[t.sensors == n], lab).test_rise_std.iloc[0] for n in (5, 8, 10)]
            xs = np.arange(3) + (k - 1) * w
            ax.bar(xs, v, w * 0.92, color=col, label=lab, yerr=e, capsize=2, error_kw={"lw": 0.8})
            for x, y in zip(xs, v):
                ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(3)); ax.set_xticklabels(["5 sensors", "8 sensors", "10 sensors"])
        ax.set_title({"i5": "i5-3337U", "i7": "i7-8650U"}[proc], loc="left", fontsize=10)
    axes[0].set_ylabel("Relative error vs temperature rise\n(test frames, lower is better)"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Real thermal maps, full training data: DeepONet vs Ruchika's PINN vs hybrid (mean of 5 seeds)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "results_full.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # chart 2: few training workloads (i7)
    if len(exps) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharey=True)
        order = [e for e in ("low1", "low2", "low4", "full") if e in exps]
        xl = {"low1": 1, "low2": 2, "low4": 4, "full": 13}
        for ax, n in zip(axes, (5, 8, 10)):
            for lab, col, mk in [("DeepONet", BLUE, "o"), ("PINN", ORANGE, None), ("Hybrid (validation-chosen)", AQUA, "s")]:
                ys = [pick(T[(T.experiment == e) & (T.processor == "i7") & (T.sensors == n)], lab).test_rise.iloc[0] for e in order]
                xs = [xl[e] for e in order]
                if lab == "PINN":
                    ax.plot(xs, ys, color=col, lw=2, label=lab)
                else:
                    ax.plot(xs, ys, color=col, lw=2, marker=mk, label=lab)
            ax.set_ylim(0, 1.0)
            d1 = pick(T[(T.experiment == order[0]) & (T.processor == "i7") & (T.sensors == n)], "DeepONet").test_rise.iloc[0]
            if d1 > 1.0:
                ax.annotate(f"DeepONet: {d1:.1f} (off scale)", (xl[order[0]], 1.0), xytext=(xl[order[0]] * 1.15, 0.86), fontsize=7.5, color=BLUE)
            ax.set_xscale("log"); ax.set_xticks([xl[e] for e in order]); ax.set_xticklabels([str(xl[e]) for e in order]); ax.minorticks_off()
            ax.set_title(f"{n} sensors", loc="left", fontsize=10); ax.set_xlabel("Training workloads (i7, 13 = all)")
        axes[0].set_ylabel("Relative error vs temperature rise\n(test frames)"); axes[2].legend(frameon=False, fontsize=8)
        fig.suptitle("i7: error as the DeepONet is trained on fewer workloads (the PINN uses none)", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(); fig.savefig(R / "results_few_workloads.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
