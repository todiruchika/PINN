"""Train Model C: physics-informed DeepONet (Model A + heat-equation residual).

Identical data, split, scaling, sensors, validation rule and metrics to Model A
(``train_deeponet.py``). The only change is the loss:

    L = L_data + lambda_pde * mean(R^2)  [+ lambda_q * L_source]

* ``L_data``  : MSE of predicted vs true temperature field at random grid points.
* ``R``       : strong-form residual  -eps div(k(T) grad theta) + theta - q~  of the
                predicted field at random *interior* collocation points, with the
                source q~ predicted by the network itself (non-negative, leakage
                applied on top). No power-map labels are used by default.
* ``L_source``: optional (``lambda_q > 0``, off by default) supervision of the
                predicted source with the dataset's ground-truth power map. This
                uses extra training labels, so it is an ablation, not the default.

Usage
-----
    python -m src.train.train_hybrid --config configs/hybrid.yaml
    python -m src.train.train_hybrid --sensors 5 --seeds 0 --epochs 20   # smoke test
    python -m src.train.train_hybrid --sensors 5 --seeds 0 --lambda-pde 0  # physics off
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import GRID, RAW_DIR, load_metadata
from src.data.splits import load_splits
from src.eval.metrics import all_metrics
from src.models.hybrid import build_hybrid
from src.physics.residual import relative_pde_residual
from src.physics.losses import ChipContext, Physics, pde_residual, sample_interior
from src.train.train_deeponet import (
    DTYPE, branch_inputs, fields_norm, grid_points, load_split, save_map_figure, set_seed,
)

RESULTS_DIR = Path("results/model_c")


def source_rel_l2(q_dyn_pred, T_pred_C, Q_true, physics: Physics, scale: float) -> float:
    """Relative L2 between the predicted total source and the dataset's ``Q`` (scoring only)."""
    total = q_dyn_pred * physics.leakage_factor(torch.as_tensor(T_pred_C)).numpy()
    target = Q_true / (physics.h_eff * scale)
    return float(np.linalg.norm(total - target) / (np.linalg.norm(target) + 1e-12))


def train_one_config(n_sensors: int, seed: int, config: dict, data: dict,
                     quiet: bool = True, save_maps: int = 0) -> dict:
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    tr = config["training"]
    physics = Physics.from_config(config.get("physics"))
    lam_pde, lam_q = float(tr["lambda_pde"]), float(tr.get("lambda_q", 0.0))
    ramp = int(tr.get("pde_ramp_epochs", 0))
    n_colloc = int(tr["n_colloc"])

    tr_d, va_d, te_d = data["train"], data["validation"], data["test"]
    scale = float((tr_d["T"] - tr_d["T_amb"][:, None, None]).max())

    train_b = torch.tensor(branch_inputs(tr_d, scale), dtype=DTYPE, device=device)
    train_f = torch.tensor(fields_norm(tr_d, scale), dtype=DTYPE, device=device)
    train_Q = torch.tensor((tr_d["Q"] / (physics.h_eff * scale)).reshape(len(tr_d["ids"]), -1),
                           dtype=DTYPE, device=device)
    train_Tamb = torch.tensor(tr_d["T_amb"], dtype=DTYPE, device=device)
    val_b = torch.tensor(branch_inputs(va_d, scale), dtype=DTYPE, device=device)
    grid_xy = torch.tensor(grid_points(), dtype=DTYPE, device=device)

    model = build_hybrid({"n_sensors": n_sensors, "net": config["model"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    epochs = int(tr["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tr["lr_min"]))
    batch_size, n_query = int(tr["batch_size"]), int(tr["n_query"])
    eval_every, patience = int(tr["eval_every"]), int(tr["patience"])

    gen = torch.Generator().manual_seed(seed)
    n_train, n_pts = train_b.shape[0], grid_xy.shape[0]

    def val_error() -> float:
        model.eval()
        with torch.no_grad():
            theta, _ = model(val_b, grid_xy)
            pred = theta.cpu().numpy().reshape(-1, GRID, GRID) * scale + va_d["T_amb"][:, None, None]
        errs = []
        for i in range(len(va_d["ids"])):
            m = va_d["mask"][i]
            errs.append(np.linalg.norm(pred[i][m] - va_d["T"][i][m]) / np.linalg.norm(va_d["T"][i][m]))
        return float(np.mean(errs))

    best = {"val": float("inf"), "state": None, "epoch": -1}
    bad = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        w_pde = lam_pde * (min(1.0, (epoch + 1) / ramp) if ramp > 0 else 1.0)
        perm = torch.randperm(n_train, generator=gen)
        tot = {"data": 0.0, "pde": 0.0, "src": 0.0}
        nb = 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            B = len(idx)
            q_idx = torch.randint(0, n_pts, (n_query,), generator=gen)
            theta, q_dyn = model(train_b[idx], grid_xy[q_idx])
            loss_data = torch.mean((theta - train_f[idx][:, q_idx]) ** 2)
            loss = loss_data
            loss_pde = loss_src = torch.zeros(())

            if w_pde > 0:
                # independent interior points per chip; each row carries its chip's embedding
                xy_c = sample_interior(B * n_colloc, generator=gen).to(device=device, dtype=DTYPE)
                b_rep = model.embed_branch(train_b[idx]).repeat_interleave(n_colloc, dim=0)
                ctx = ChipContext(T_amb=train_Tamb[idx].repeat_interleave(n_colloc)[:, None], scale=scale)
                res = pde_residual(
                    lambda xy: model.theta_pointwise(b_rep, xy),
                    lambda xy, Tc: model.q_pointwise(b_rep, xy) * physics.leakage_factor(Tc),
                    xy_c, physics, ctx,
                )
                loss_pde = torch.mean(res ** 2)
                loss = loss + w_pde * loss_pde

            if lam_q > 0:
                T_c = train_Tamb[idx][:, None] + scale * theta.detach()
                q_tot = q_dyn * physics.leakage_factor(T_c)
                loss_src = torch.mean((q_tot - train_Q[idx][:, q_idx]) ** 2)
                loss = loss + lam_q * loss_src

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot["data"] += float(loss_data.detach())
            tot["pde"] += float(loss_pde.detach())
            tot["src"] += float(loss_src.detach())
            nb += 1
        sched.step()
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            v = val_error()
            if not quiet:
                print(f"[N={n_sensors} seed={seed}] epoch {epoch + 1}/{epochs} data={tot['data'] / nb:.5f} "
                      f"pde={tot['pde'] / nb:.5f} src={tot['src'] / nb:.5f} val_rel_l2={v:.5f}", flush=True)
            if v < best["val"]:
                best = {"val": v, "epoch": epoch + 1,
                        "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best["state"])

    model.eval()
    test_b = torch.tensor(branch_inputs(te_d, scale), dtype=DTYPE, device=device)
    rows, maps = [], []
    with torch.no_grad():
        for i, sid in enumerate(te_d["ids"]):
            t1 = time.perf_counter()
            th, qd = model.predict_field(test_b[i], grid_xy)
            inf_s = time.perf_counter() - t1
            pred = th.cpu().numpy().reshape(GRID, GRID) * scale + te_d["T_amb"][i]
            q_pred = qd.cpu().numpy().reshape(GRID, GRID)
            rows.append({"sample_id": sid, "n_sensors": n_sensors, "seed": seed,
                         **all_metrics(pred, te_d["T"][i], te_d["mask"][i]),
                         "pde_residual_rel": relative_pde_residual(pred, te_d["Q"][i], te_d["T_amb"][i]),
                         "source_rel_l2": source_rel_l2(q_pred, pred, te_d["Q"][i], physics, scale),
                         "inference_seconds": inf_s})
            if i < save_maps:
                maps.append((sid, pred, te_d["T"][i], te_d["pos"][i]))
    return {"rows": rows, "maps": maps, "best_val_rel_l2": best["val"], "best_epoch": best["epoch"],
            "train_seconds": train_seconds, "model": model,
            "n_params": sum(p.numel() for p in model.parameters())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model C (physics-informed DeepONet)")
    ap.add_argument("--config", default="configs/hybrid.yaml")
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lambda-pde", type=float, default=None, help="physics weight (0 = physics off)")
    ap.add_argument("--lambda-q", type=float, default=None, help="source-label weight (ablation)")
    ap.add_argument("--hard-bc", type=int, default=None, help="1/0 override of model.hard_bc")
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--save-maps", type=int, default=0)
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lambda_pde is not None:
        config["training"]["lambda_pde"] = args.lambda_pde
    if args.lambda_q is not None:
        config["training"]["lambda_q"] = args.lambda_q
    if args.hard_bc is not None:
        config["model"]["hard_bc"] = bool(args.hard_bc)
    sensors = [int(s) for s in args.sensors.split(",")] if args.sensors else config["n_sensors"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config["seeds"]
    strategy = args.strategy or config["sensor_strategy"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = config.get("name", "hybrid")

    meta = load_metadata(RAW_DIR)
    splits = load_splits(RAW_DIR / "splits.csv")
    hw = "CPU" if config.get("device", "cpu") == "cpu" else torch.cuda.get_device_name(0)
    print("splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) +
          f"; strategy={strategy}; hw={hw}; lambda_pde={config['training']['lambda_pde']}; "
          f"lambda_q={config['training'].get('lambda_q', 0)}; hard_bc={config['model'].get('hard_bc', True)}")

    seed_rows = {s: [] for s in seeds}
    for n in sensors:
        data = {sp: load_split(splits[sp], n, strategy, meta) for sp in ("train", "validation", "test")}
        for seed in seeds:
            print(f"=== Model C ({tag}): N={n}, seed={seed} ===", flush=True)
            res = train_one_config(n, seed, config, data, quiet=args.quiet, save_maps=args.save_maps)
            df = pd.DataFrame(res["rows"])
            df.to_csv(out_dir / f"per_sample_n{n}_seed{seed}.csv", index=False)
            summary = {"config": f"{tag}_{strategy}_n{n}", "seed": seed,
                       **df.drop(columns=["sample_id", "n_sensors", "seed"]).mean().to_dict(),
                       "best_val_rel_l2": res["best_val_rel_l2"], "best_epoch": res["best_epoch"],
                       "n_params": res["n_params"], "train_hours": res["train_seconds"] / 3600}
            seed_rows[seed].append(summary)
            pd.DataFrame(seed_rows[seed]).to_csv(out_dir / f"seed_{seed}.csv", index=False)
            print(f"    best val rel_l2={res['best_val_rel_l2']:.4f} (epoch {res['best_epoch']}) "
                  f"-> test rel_l2={summary['relative_l2']:.4f}, MAE={summary['mae_C']:.2f} C, "
                  f"RMSE={summary['rmse_C']:.2f} C, source_rel_l2={summary['source_rel_l2']:.3f}, "
                  f"params={res['n_params']}, train={res['train_seconds']:.0f}s", flush=True)
            if res["maps"]:
                print("    saved", save_map_figure(res["maps"], n, seed, out_dir))
            if args.save_checkpoints:
                ck = out_dir / "checkpoints"
                ck.mkdir(exist_ok=True)
                torch.save(res["model"].state_dict(), ck / f"{tag}_n{n}_seed{seed}.pt")


if __name__ == "__main__":
    main()
