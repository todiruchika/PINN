"""Hybrid of the DeepONet and a physics-based (heat-equation) reconstruction on the real thermal maps.

Three models are scored on the same test frames and the same noisy sensor readings:

* **DeepONet**: as in ``train_real_deeponet.py`` (5 seeds).
* **Physics**: the screened-Poisson / heat-equation reconstruction of ``src/physics/screened_poisson.py`` (PINN-style inverse problem,
  uses no training maps). Its three settings (decay length, source-bump spacing, ridge) are chosen on validation frames.
* **Hybrids** (all settings chosen on validation frames only, test frames scored once):
    - ``blend``:    ``w * DeepONet + (1 - w) * Physics`` with ``w`` in {0, 0.1, ..., 1};
    - ``correct``:  ``DeepONet + physics response fitted to the DeepONet's residual at the sensors``.
* Reference: gappy POD and the training-mean map (classical, from ``train_real_deeponet.baselines``).

    python -m src.eval.real_hybrid --processor i5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.physics.screened_poisson import build_response, reconstruct
from src.train.train_real_deeponet import baselines, metrics_row, prepare, read_sensors, train_one

ELLS = [4, 8, 16, 32, 64]
SPACINGS = [8, 14]
LAMS = [1e-4, 1e-2, 1e-1, 1.0, 10.0]
WS = np.round(np.arange(0, 1.0001, 0.1), 1)
RESULTS = Path("results/real_hybrid")


def rise_err(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """Mean over frames of ||pred - true|| / ||true - min(true)|| on non-sensor pixels (the validation selection score)."""
    return float(np.mean([np.linalg.norm((p - t)[mask]) / np.linalg.norm((t - t.min())[mask]) for p, t in zip(pred, true)]))


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepONet + physics hybrid on real thermal maps")
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--processor", default="i5", choices=["i5", "i7"])
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--val-stride", type=int, default=3, help="use every k-th validation frame for tuning (speed)")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps:
        cfg["training"]["steps"] = a.steps
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    out = RESULTS / a.processor; out.mkdir(parents=True, exist_ok=True)

    D = prepare(a.processor, cfg)
    H, W = D["shape"]
    print(f"{a.processor}: map {D['shape']}, frames " + ", ".join(f"{k}={len(v)}" for k, v in D["maps"].items()), flush=True)
    Phis = {(e, s): build_response(D["shape"], e, s) for e in ELLS for s in SPACINGS}
    vi = np.arange(0, len(D["maps"]["validation"]), a.val_stride)
    vtrue = D["maps"]["validation"]
    runs, frames = [], []

    for n in sensors:
        readings = {}
        for sid, k in enumerate(("train", "validation", "test")):
            first = D["frames"][k][0][1] if k != "train" and len({w for w, _ in D["frames"][k]}) == 1 else 0
            readings[k] = read_sensors(D["maps"][k], D["rc"], first, n, cfg, sid)
        flat = D["rc"][:n, 0] * W + D["rc"][:n, 1]
        mask = np.ones((H, W), bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
        true_test = D["maps"]["test"]

        # --- physics only: choose (ell, spacing, ridge) on validation frames ----------------------------------------
        def physics_maps(split, hp, idx=None):
            e, s, lam = hp
            r = readings[split] if idx is None else readings[split][idx]
            return np.stack([reconstruct(Phis[(e, s)], flat, y, lam).reshape(H, W) for y in r])

        grid = [(e, s, l) for e in ELLS for s in SPACINGS for l in LAMS]
        hp_phys = min(grid, key=lambda hp: rise_err(physics_maps("validation", hp, vi), vtrue[vi], mask))
        P_val = physics_maps("validation", hp_phys); P_test = physics_maps("test", hp_phys)
        print(f"[{a.processor} N={n}] physics settings (ell, spacing, ridge) = {hp_phys}", flush=True)

        base = pd.DataFrame(baselines(D, n, cfg, readings))
        for m, g in base.groupby("model"):
            runs.append({"n_sensors": n, "seed": -1, "model": m, **g.drop(columns=["model", "frame"]).mean().to_dict()})

        for seed in seeds:
            rows, best, secs, _, _, preds = train_one(D, n, seed, cfg, readings, quiet=True, return_preds=True)
            Dv, Dt = preds["validation"], preds["test"]

            # blend weight on validation
            w_best = min(WS, key=lambda w: rise_err((w * Dv + (1 - w) * P_val)[vi], vtrue[vi], mask))
            B_test = w_best * Dt + (1 - w_best) * P_test

            # correction settings on validation
            def corrected(split, hp, base_maps, idx=None):
                e, s, lam = hp
                r = readings[split] if idx is None else readings[split][idx]
                b = base_maps if idx is None else base_maps[idx]
                return np.stack([reconstruct(Phis[(e, s)], flat, y, lam, base=bm.ravel()).reshape(H, W) for y, bm in zip(r, b)])
            hp_corr = min(grid, key=lambda hp: rise_err(corrected("validation", hp, Dv, vi), vtrue[vi], mask))
            C_val, C_test = corrected("validation", hp_corr, Dv), corrected("test", hp_corr, Dt)

            # blend first, then correct the blend
            Bv = w_best * Dv + (1 - w_best) * P_val
            hp_both = min(grid, key=lambda hp: rise_err(corrected("validation", hp, Bv, vi), vtrue[vi], mask))
            BC_val, BC_test = corrected("validation", hp_both, Bv), corrected("test", hp_both, B_test)

            ALL = {"DeepONet": (Dv, Dt), "Physics": (P_val, P_test), "Hybrid 1: blend": (Bv, B_test),
                   "Hybrid 2: DeepONet + physics correction": (C_val, C_test),
                   "Hybrid 3: blend + physics correction": (BC_val, BC_test)}
            for name, (pv, pt) in ALL.items():
                mr = pd.DataFrame([metrics_row(pt[k], true_test[k], mask) for k in range(len(true_test))])
                runs.append({"n_sensors": n, "seed": seed, "model": name, **mr.mean().to_dict(),
                             "val_rise_error": rise_err(pv, vtrue, mask),
                             "blend_w": w_best if "blend" in name else np.nan,
                             "corr_settings": str(hp_corr if name.startswith("Hybrid 2") else hp_both if name.startswith("Hybrid 3") else "")})
                frames += [{"n_sensors": n, "seed": seed, "model": name, "frame": k, "rel_l2_rise": mr.rel_l2_rise[k]} for k in range(len(mr))]
            last = {r["model"]: r for r in runs[-5:]}
            print(f"    seed {seed}  val rise error / test rise error: " + " | ".join(
                f"{m.split(':')[0]} {r['val_rise_error']:.3f}/{r['rel_l2_rise']:.3f}" for m, r in last.items()) + f"  ({secs:.0f}s)", flush=True)
            pd.DataFrame(runs).to_csv(out / "summary_runs.csv", index=False)
            pd.DataFrame(frames).to_csv(out / "per_frame_rise_error.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "summary_runs.csv", index=False)
    pd.DataFrame(frames).to_csv(out / "per_frame_rise_error.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
