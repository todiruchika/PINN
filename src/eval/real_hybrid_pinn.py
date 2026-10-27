"""Hybrid of the DeepONet and Ruchika's per-chip PINN on the real thermal maps.

Stage ``deeponet``: train the DeepONet (5 seeds x 3 sensor counts, same recipe as ``train_real_deeponet``) and save its predictions on the
sampled validation and test frames (``subset_idx``), next to the PINN predictions made by ``real_pinn_runs.py`` on the same frames
with the same sensor readings.

Stage ``blend``: for every (processor, sensor count, seed), with all settings chosen on the validation frames only,
  * DeepONet, PINN (the two individual models);
  * Hybrid A, blend:            ``w * DeepONet + (1 - w) * PINN``;
  * Hybrid B, correction:       DeepONet + heat-equation response fitted to the DeepONet's residual at the sensors
                                (``src/physics/screened_poisson.py``; no PINN involved);
  * Hybrid C, blend + correction.
"The hybrid" is the single variant with the lowest mean validation error over all settings.

    python -m src.eval.real_hybrid_pinn --stage deeponet --processor i5
    python -m src.eval.real_hybrid_pinn --stage blend
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.real_hybrid import ELLS, LAMS, SPACINGS, WS, rise_err
from src.eval.real_pinn_runs import OUT as PINN_OUT, SUBSET, subset_idx
from src.physics.screened_poisson import build_response, reconstruct
from src.train.train_real_deeponet import metrics_row, prepare, read_sensors, train_one

OUT = Path("results/real_pinn_hybrid")
NAMES = {"i5": "i5-3337U", "i7": "i7-8650U"}


def restrict_training(D: dict, k: int, seed: int = 123) -> list:
    """Keep only ``k`` (seeded, nested) of the training workloads. Sensor layout stays as fitted on all training workloads."""
    names = sorted({w for w, _ in D["frames"]["train"]})
    chosen = sorted(np.random.default_rng(seed).permutation(names)[:k].tolist())
    keep = [j for j, (w, _) in enumerate(D["frames"]["train"]) if w in chosen]
    D["maps"]["train"] = D["maps"]["train"][keep]
    D["mu"], D["sd"] = float(D["maps"]["train"].mean()), float(D["maps"]["train"].std())
    return chosen


def stage_deeponet(proc: str, cfg: dict, sensors: list, seeds: list, k_workloads: int | None = None) -> None:
    D = prepare(proc, cfg)
    tag = proc
    if k_workloads:
        chosen = restrict_training(D, k_workloads)
        tag = f"{proc}_low{k_workloads}"
        print(f"{tag}: training workloads = {chosen} ({len(D['maps']['train'])} frames)", flush=True)
    out = OUT / tag; out.mkdir(parents=True, exist_ok=True)
    idx = {s: subset_idx(len(D["maps"][s]), SUBSET[proc]) for s in ("validation", "test")}
    for n in sensors:
        readings = {}
        for sid, k in enumerate(("train", "validation", "test")):
            first = D["frames"][k][0][1] if k != "train" and len({w for w, _ in D["frames"][k]}) == 1 else 0
            readings[k] = read_sensors(D["maps"][k], D["rc"], first, n, cfg, sid)
        for seed in seeds:
            f = out / f"deeponet_n{n}_seed{seed}.npz"
            if f.exists():
                continue
            _, best, secs, _, _, preds = train_one(D, n, seed, cfg, readings, quiet=True, return_preds=True)
            np.savez_compressed(f, val_idx=idx["validation"], test_idx=idx["test"],
                                val_pred=preds["validation"][idx["validation"]].astype(np.float32),
                                test_pred=preds["test"][idx["test"]].astype(np.float32))
            print(f"{proc} N={n} seed={seed}: saved ({secs:.0f}s, val {best['v']:.4f})", flush=True)


def stage_blend(cfg: dict, sensors: list, seeds: list, k_workloads: int | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs, frames = [], []
    suffix = f"_low{k_workloads}" if k_workloads else ""
    for proc in (["i7"] if k_workloads else NAMES):
        D = prepare(proc, cfg)
        if k_workloads:
            restrict_training(D, k_workloads)
        H, W = D["shape"]
        Phis = {(e, s): build_response(D["shape"], e, s) for e in ELLS for s in SPACINGS}
        grid = [(e, s, l) for e in ELLS for s in SPACINGS for l in LAMS]
        for n in sensors:
            readings = {}
            for sid, k in enumerate(("train", "validation", "test")):
                first = D["frames"][k][0][1] if k != "train" and len({w for w, _ in D["frames"][k]}) == 1 else 0
                readings[k] = read_sensors(D["maps"][k], D["rc"], first, n, cfg, sid)
            flat = D["rc"][:n, 0] * W + D["rc"][:n, 1]
            mask = np.ones((H, W), bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
            first_file = np.load(OUT / (proc + suffix) / f"deeponet_n{n}_seed{seeds[0]}.npz")
            vi, ti = first_file["val_idx"], first_file["test_idx"]
            vtrue, ttrue = D["maps"]["validation"][vi], D["maps"]["test"][ti]
            P_val = np.stack([np.load(PINN_OUT / proc / f"validation_{i:03d}_N{n}.npy") for i in vi]).astype(np.float64)
            P_test = np.stack([np.load(PINN_OUT / proc / f"test_{i:03d}_N{n}.npy") for i in ti]).astype(np.float64)
            rv, rt = readings["validation"][vi], readings["test"][ti]

            def corrected(r, base, hp):
                e, s, lam = hp
                return np.stack([reconstruct(Phis[(e, s)], flat, y, lam, base=b.ravel()).reshape(H, W) for y, b in zip(r, base)])

            for seed in seeds:
                z = np.load(OUT / (proc + suffix) / f"deeponet_n{n}_seed{seed}.npz")
                Dv, Dt = z["val_pred"].astype(np.float64), z["test_pred"].astype(np.float64)
                w = min(WS, key=lambda w_: rise_err(w_ * Dv + (1 - w_) * P_val, vtrue, mask))
                Bv, Bt = w * Dv + (1 - w) * P_val, w * Dt + (1 - w) * P_test
                hp_c = min(grid, key=lambda hp: rise_err(corrected(rv, Dv, hp), vtrue, mask))
                Cv, Ct = corrected(rv, Dv, hp_c), corrected(rt, Dt, hp_c)
                hp_bc = min(grid, key=lambda hp: rise_err(corrected(rv, Bv, hp), vtrue, mask))
                BCv, BCt = corrected(rv, Bv, hp_bc), corrected(rt, Bt, hp_bc)
                allm = {"DeepONet": (Dv, Dt), "PINN": (P_val, P_test), "Hybrid A: blend (DeepONet, PINN)": (Bv, Bt),
                        "Hybrid B: DeepONet + physics correction": (Cv, Ct), "Hybrid C: blend + physics correction": (BCv, BCt)}
                for name, (pv, pt) in allm.items():
                    mr = pd.DataFrame([metrics_row(pt[k], ttrue[k], mask) for k in range(len(ttrue))])
                    runs.append({"processor": proc, "n_sensors": n, "seed": seed, "model": name, **mr.mean().to_dict(),
                                 "val_rise_error": rise_err(pv, vtrue, mask), "blend_w": w if "blend" in name else np.nan})
                    frames += [{"processor": proc, "n_sensors": n, "seed": seed, "model": name, "frame": int(ti[k]), "rel_l2_rise": mr.rel_l2_rise[k]} for k in range(len(mr))]
            print(f"{proc} N={n} done", flush=True)
    pd.DataFrame(runs).to_csv(OUT / f"summary_runs{suffix}.csv", index=False)
    pd.DataFrame(frames).to_csv(OUT / f"per_frame_rise_error{suffix}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["deeponet", "blend"])
    ap.add_argument("--processor", choices=list(NAMES))
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--k-workloads", type=int, default=None, help="low-data experiment (i7): train on only this many workloads")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    if a.stage == "deeponet":
        stage_deeponet(a.processor, cfg, sensors, seeds, a.k_workloads)
    else:
        stage_blend(cfg, sensors, seeds, a.k_workloads)


if __name__ == "__main__":
    main()
