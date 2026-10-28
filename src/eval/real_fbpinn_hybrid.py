"""Hybrid of the DeepONet and the per-frame FBPINN on the real thermal maps (same recipe as ``real_hybrid_pinn.stage_blend``).

Inputs: the DeepONet predictions of ``real_hybrid_pinn --stage deeponet`` (``results/real_pinn_hybrid/<tag>/deeponet_n*_seed*.npz``,
git-ignored) and the FBPINN predictions of ``real_fbpinn_runs.py`` on the same sampled frames with the same readings. For every
(processor, sensor count, DeepONet seed), with all settings chosen on the validation frames only:
  * DeepONet, FBPINN (the two individual models);
  * Hybrid A, blend:            ``w * DeepONet + (1 - w) * FBPINN``;
  * Hybrid B, correction:       DeepONet + heat-equation response fitted to its residual at the sensors (no FBPINN involved, so
                                identical to Hybrid B of ``real_hybrid_pinn``);
  * Hybrid C, blend + correction.
The FBPINN uses no training maps, so the few-training-workloads experiments reuse the same FBPINN solves.
A setting is skipped (and reported as not run) unless every sampled validation and test frame has an FBPINN solve.

    python -m src.eval.real_fbpinn_hybrid                      # full training data, i5 and i7
    python -m src.eval.real_fbpinn_hybrid --k-workloads 4      # i7, 4 training workloads
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.real_hybrid import ELLS, LAMS, SPACINGS, WS, rise_err
from src.eval.real_hybrid_pinn import OUT as DEEPONET_OUT, restrict_training
from src.eval.real_fbpinn_runs import OUT, PREDS
from src.physics.screened_poisson import build_response, reconstruct
from src.train.train_real_deeponet import metrics_row, prepare
from src.train.train_real_fno import split_readings

NAMES = {"i5": "i5-3337U", "i7": "i7-8650U"}
NEW = "FBPINN"
HYB = [f"Hybrid A: blend (DeepONet, {NEW})", "Hybrid B: DeepONet + physics correction", "Hybrid C: blend + physics correction"]


def stage_blend(cfg: dict, sensors: list, seeds: list, k_workloads: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs, frames, missing = [], [], []
    suffix = f"_low{k_workloads}" if k_workloads else ""
    for proc in (["i7"] if k_workloads else NAMES):
        D = prepare(proc, cfg)
        if k_workloads:
            restrict_training(D, k_workloads)
        H, W = D["shape"]
        Phis = {(e, s): build_response(D["shape"], e, s) for e in ELLS for s in SPACINGS}
        grid = [(e, s, l) for e in ELLS for s in SPACINGS for l in LAMS]
        for n in sensors:
            readings = split_readings(D, n, cfg)
            flat = D["rc"][:n, 0] * W + D["rc"][:n, 1]
            mask = np.ones((H, W), bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
            first_file = np.load(DEEPONET_OUT / (proc + suffix) / f"deeponet_n{n}_seed{seeds[0]}.npz")
            vi, ti = first_file["val_idx"], first_file["test_idx"]
            need = [PREDS / proc / f"validation_{i:03d}_N{n}.npy" for i in vi] + [PREDS / proc / f"test_{i:03d}_N{n}.npy" for i in ti]
            if not all(f.exists() for f in need):
                missing.append({"processor": proc, "n_sensors": n, "solved": sum(f.exists() for f in need), "needed": len(need)})
                print(f"{proc} N={n}: FBPINN incomplete ({missing[-1]['solved']}/{len(need)} frames), setting not run", flush=True)
                continue
            vtrue, ttrue = D["maps"]["validation"][vi], D["maps"]["test"][ti]
            M_val = np.stack([np.load(f) for f in need[:len(vi)]]).astype(np.float64)
            M_test = np.stack([np.load(f) for f in need[len(vi):]]).astype(np.float64)
            rv, rt = readings["validation"][vi], readings["test"][ti]

            def corrected(r, base, hp):
                e, s, lam = hp
                return np.stack([reconstruct(Phis[(e, s)], flat, y, lam, base=b.ravel()).reshape(H, W) for y, b in zip(r, base)])

            for seed in seeds:
                z = np.load(DEEPONET_OUT / (proc + suffix) / f"deeponet_n{n}_seed{seed}.npz")
                Dv, Dt = z["val_pred"].astype(np.float64), z["test_pred"].astype(np.float64)
                w = min(WS, key=lambda w_: rise_err(w_ * Dv + (1 - w_) * M_val, vtrue, mask))
                Bv, Bt = w * Dv + (1 - w) * M_val, w * Dt + (1 - w) * M_test
                hp_c = min(grid, key=lambda hp: rise_err(corrected(rv, Dv, hp), vtrue, mask))
                Cv, Ct = corrected(rv, Dv, hp_c), corrected(rt, Dt, hp_c)
                hp_bc = min(grid, key=lambda hp: rise_err(corrected(rv, Bv, hp), vtrue, mask))
                BCv, BCt = corrected(rv, Bv, hp_bc), corrected(rt, Bt, hp_bc)
                allm = {"DeepONet": (Dv, Dt), NEW: (M_val, M_test), HYB[0]: (Bv, Bt), HYB[1]: (Cv, Ct), HYB[2]: (BCv, BCt)}
                for name, (pv, pt) in allm.items():
                    mr = pd.DataFrame([metrics_row(pt[k], ttrue[k], mask) for k in range(len(ttrue))])
                    runs.append({"processor": proc, "n_sensors": n, "seed": seed, "model": name, **mr.mean().to_dict(),
                                 "val_rise_error": rise_err(pv, vtrue, mask), "blend_w": w if "blend" in name else np.nan})
                    frames += [{"processor": proc, "n_sensors": n, "seed": seed, "model": name, "frame": int(ti[k]),
                                "rel_l2_rise": mr.rel_l2_rise[k]} for k in range(len(mr))]
            print(f"{proc} N={n} done", flush=True)
    pd.DataFrame(runs).to_csv(OUT / f"summary_runs{suffix}.csv", index=False)
    pd.DataFrame(frames).to_csv(OUT / f"per_frame_rise_error{suffix}.csv", index=False)
    pd.DataFrame(missing, columns=["processor", "n_sensors", "solved", "needed"]).to_csv(OUT / f"not_run{suffix}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--k-workloads", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    stage_blend(cfg, sensors, seeds, a.k_workloads)


if __name__ == "__main__":
    main()
