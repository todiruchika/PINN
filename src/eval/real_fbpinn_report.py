"""Summarise results/real_fbpinn: DeepONet vs FBPINN vs hybrids, with Ruchika's PINN, its hybrid, gappy POD and the training-mean map
scored on the same sampled test frames (layout of ``real_hybrid_pinn_report``).

In each experiment "the hybrid" is the one variant (A blend, B correction, C blend + correction) with the lowest mean *validation*
error over that experiment's settings and seeds; test frames play no part in that choice. Paired 95% intervals: bootstrap over the
sampled test frames of the per-frame error (mean over seeds), ``gain = reference - model`` (positive = model better).
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.eval.real_fbpinn_hybrid import HYB, NEW
from src.train.train_real_deeponet import baselines, prepare
from src.train.train_real_fno import split_readings

R = Path("results/real_fbpinn")
PH = Path("results/real_pinn_hybrid")
rng = np.random.default_rng(0)


def references(cfg: dict) -> pd.DataFrame:
    """Per-frame rise error of the existing models on the sampled test frames (full training data)."""
    old = pd.read_csv(PH / "per_frame_rise_error.csv")
    old = old[old.model.isin(["PINN", "Hybrid B: DeepONet + physics correction"])].copy()
    old["model"] = old.model.map({"PINN": "PINN (Ruchika)", "Hybrid B: DeepONet + physics correction": "existing hybrid (DeepONet + PINN study)"})
    rows = []
    for proc in ("i5", "i7"):
        D = prepare(proc, cfg)
        keep = set(old[old.processor == proc].frame)
        for n in cfg["sensor_counts"]:
            for r in baselines(D, n, cfg, split_readings(D, n, cfg)):
                if r["frame"] in keep:
                    rows.append({"processor": proc, "n_sensors": n, "seed": -1, "model": r["model"], "frame": r["frame"], "rel_l2_rise": r["rel_l2_rise"]})
    return pd.concat([old, pd.DataFrame(rows)], ignore_index=True)


def paired(f: pd.DataFrame, model: str, ref: str) -> dict:
    d = (f[ref] - f[model]).dropna().values
    bs = [rng.choice(d, len(d)).mean() for _ in range(3000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"test_frames": len(d), "mean_gain": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "verdict": "better" if lo > 0 else ("worse" if hi < 0 else "tie")}


def costs() -> pd.DataFrame:
    fb = pd.concat(pd.read_csv(f) for f in glob.glob(str(R / "*" / "metrics_shard*.csv")))
    pn = pd.concat(pd.read_csv(f) for f in glob.glob(str(Path("results/real_pinn") / "*" / "metrics_shard*.csv")))
    dn = pd.concat(pd.read_csv(f"results/real_deeponet/{p}/summary_runs.csv") for p in ("i5", "i7"))
    dn = dn[dn.model == "DeepONet"]
    return pd.DataFrame([
        {"model": "DeepONet", "kind": "trained once per (processor, sensors, seed)", "n_params": int(dn.n_params.median()),
         "seconds_median": dn.train_seconds.median(), "runs": len(dn)},
        {"model": "PINN (Ruchika)", "kind": "per frame, 600 Adam + 100 L-BFGS", "n_params": int(pn.n_params.median()),
         "seconds_median": pn.train_seconds.median(), "runs": len(pn)},
        {"model": NEW, "kind": "per frame, 600 Adam + 100 L-BFGS", "n_params": int(fb.n_params.median()),
         "seconds_median": fb.train_seconds.median(), "runs": len(fb)}])


def main() -> None:
    cfg = yaml.safe_load(open("configs/real_deeponet.yaml"))
    exps = ["full"] + [f"low{k}" for k in (4, 2) if (R / f"summary_runs_low{k}.csv").exists()]
    rows, pairs = [], []
    ref_frames = references(cfg)
    for exp in exps:
        sfx = "" if exp == "full" else f"_{exp}"
        runs, fr = pd.read_csv(R / f"summary_runs{sfx}.csv"), pd.read_csv(R / f"per_frame_rise_error{sfx}.csv")
        if runs.empty:
            continue
        vm = runs.groupby("model").val_rise_error.mean()
        best = vm[HYB].idxmin()
        print(f"[{exp}] mean validation error:", vm.round(3).to_dict(), "-> hybrid =", best)
        for (proc, n), g in runs.groupby(["processor", "n_sensors"]):
            m = g.groupby("model")[["rel_l2_rise", "relative_l2", "mae_C", "rmse_C", "peak_temp_error_C", "hotspot_localization_cells",
                                    "val_rise_error"]].agg(["mean", "std"])
            for name in ["DeepONet", NEW] + HYB:
                rows.append({"experiment": exp, "processor": proc, "sensors": n, "model": name, "chosen_hybrid": name == best,
                             "test_rise": m.loc[name, ("rel_l2_rise", "mean")], "test_rise_std": m.loc[name, ("rel_l2_rise", "std")],
                             "test_rel_l2": m.loc[name, ("relative_l2", "mean")], "test_mae_C": m.loc[name, ("mae_C", "mean")],
                             "test_rmse_C": m.loc[name, ("rmse_C", "mean")], "test_peak_err_C": m.loc[name, ("peak_temp_error_C", "mean")],
                             "test_hotspot_px": m.loc[name, ("hotspot_localization_cells", "mean")], "val_rise": m.loc[name, ("val_rise_error", "mean")]})
            f = fr[(fr.processor == proc) & (fr.n_sensors == n)].groupby(["model", "frame"]).rel_l2_rise.mean().unstack(0)
            if exp == "full":
                rf = ref_frames[(ref_frames.processor == proc) & (ref_frames.n_sensors == n)].groupby(["model", "frame"]).rel_l2_rise.mean().unstack(0)
                f = f.join(rf, how="left")
                for name in rf.columns:
                    rows.append({"experiment": exp, "processor": proc, "sensors": n, "model": name, "chosen_hybrid": False,
                                 "test_rise": f[name].mean(), "test_rise_std": np.nan})
            for model, refs in [(best, ["DeepONet", NEW] + (list(rf.columns) if exp == "full" else [])),
                                (NEW, ["DeepONet"] + (["PINN (Ruchika)"] if exp == "full" else []))]:
                for ref in refs:
                    pairs.append({"experiment": exp, "processor": proc, "sensors": n, "model": model, "versus": ref, **paired(f, model, ref)})
    T, C = pd.DataFrame(rows), pd.DataFrame(pairs)
    T.to_csv(R / "summary.csv", index=False); C.to_csv(R / "paired_comparison.csv", index=False)
    K = costs(); K.to_csv(R / "cost.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False)); print(); print(C.round(4).to_string(index=False)); print(); print(K.to_string(index=False))

    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "axes.axisbelow": True})
    pick = lambda t, lab: t[t.chosen_hybrid] if lab.startswith("Hybrid") else t[t.model == lab]
    labels = [("DeepONet", BLUE), ("PINN (Ruchika)", GREY), (NEW, ORANGE), ("Hybrid (validation-chosen: B, no FBPINN in it)", AQUA)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, proc in zip(axes, ["i5", "i7"]):
        t = T[(T.experiment == "full") & (T.processor == proc)]; w = 0.2
        ns = sorted(t.sensors.unique())
        for k, (lab, col) in enumerate(labels):
            v = [pick(t[t.sensors == n], lab).test_rise.iloc[0] for n in ns]
            e = [pick(t[t.sensors == n], lab).test_rise_std.iloc[0] for n in ns]
            xs = np.arange(len(ns)) + (k - 1.5) * w
            ax.bar(xs, v, w * 0.92, color=col, label=lab, yerr=np.nan_to_num(e), capsize=2, error_kw={"lw": 0.8})
            for x, y in zip(xs, v):
                ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"{n} sensors" for n in ns])
        ax.set_title(NAMES_FULL[proc], loc="left", fontsize=10)
    axes[0].set_ylabel("Relative error vs temperature rise\n(sampled test frames, lower is better)"); axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Real thermal maps: DeepONet vs per-frame PINN and FBPINN vs hybrid (DeepONet: mean of 5 seeds; PINN, FBPINN: 1 seed)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "results_full.png", dpi=170, bbox_inches="tight"); plt.close(fig)


NAMES_FULL = {"i5": "i5-3337U", "i7": "i7-8650U"}

if __name__ == "__main__":
    main()
