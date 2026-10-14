"""Train Model A (data-driven DeepONet, no physics) on data/raw/dataset.

One network per (n_sensors, seed), trained once on the train split and
evaluated on every held-out test sample, on non-sensor nodes only (PROTOCOL.md).

Input to the branch net is the sensor positions plus ``temperatures_noisy``
(the protocol's primary input); ``temperatures_true`` is never used. Fields
are nondimensionalised with the sample's known ambient and one global scale:

    theta = (T - T_amb) / dT_scale,   dT_scale = max_train (T - T_amb)

Usage
-----
    python -m src.train.train_deeponet --config configs/deeponet.yaml
    python -m src.train.train_deeponet --sensors 5 --seeds 0 --epochs 20   # smoke test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import GRID, RAW_DIR, load_metadata, load_sample
from src.data.splits import load_splits
from src.eval.metrics import all_metrics
from src.models.deeponet import build_deeponet
from src.physics.residual import relative_pde_residual
from src.sensors.readers import metre_to_idx, sensor_mask

DTYPE = torch.float32
RESULTS_DIR = Path("results/model_a")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def grid_points(n: int = GRID) -> np.ndarray:
    """``(n*n, 2)`` array of ``(x, y)`` in [0, 1]^2, row-major (row = y)."""
    ys, xs = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


def load_split(ids: list, n_sensors: int, strategy: str, meta: pd.DataFrame) -> dict:
    """Raw arrays for a split: T, T_amb, sensor positions (unit square) and noisy readings."""
    T, Q, T_amb, pos, noisy, masks = [], [], [], [], [], []
    for sid in ids:
        s = load_sample(sid, n_sensors, strategy=strategy, meta=meta)
        T.append(s["T"].astype(np.float64))
        Q.append(s["Q"].astype(np.float64))
        T_amb.append(float(s["meta"]["T_amb_C"]))
        positions_m = s["sensors"]["positions"]
        pos.append(np.array(positions_m) / 0.012)
        noisy.append(np.array(s["sensors"]["temperatures_noisy"], dtype=np.float64))
        masks.append(~sensor_mask(positions_m))
    return {"ids": ids, "T": np.stack(T), "Q": np.stack(Q), "T_amb": np.array(T_amb), "pos": np.stack(pos),
            "noisy": np.stack(noisy), "mask": np.stack(masks)}


def branch_inputs(d: dict, scale: float) -> np.ndarray:
    """``(n, 3N)``: ``[x_1, y_1, theta_1, ...]``, sensors sorted by position for a stable order."""
    n, N = d["noisy"].shape
    theta = (d["noisy"] - d["T_amb"][:, None]) / scale
    out = np.zeros((n, 3 * N), dtype=np.float32)
    for i in range(n):
        order = np.lexsort((d["pos"][i][:, 1], d["pos"][i][:, 0]))
        out[i] = np.concatenate([d["pos"][i][order], theta[i][order, None]], axis=1).ravel()
    return out


def fields_norm(d: dict, scale: float) -> np.ndarray:
    return ((d["T"] - d["T_amb"][:, None, None]) / scale).reshape(len(d["ids"]), -1).astype(np.float32)


def train_one_config(n_sensors: int, seed: int, config: dict, data: dict, quiet: bool = True, save_maps: int = 0) -> dict:
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    tr = config["training"]
    tr_d, va_d, te_d = data["train"], data["validation"], data["test"]
    scale = float((tr_d["T"] - tr_d["T_amb"][:, None, None]).max())

    train_b = torch.tensor(branch_inputs(tr_d, scale), dtype=DTYPE, device=device)
    train_f = torch.tensor(fields_norm(tr_d, scale), dtype=DTYPE, device=device)
    val_b = torch.tensor(branch_inputs(va_d, scale), dtype=DTYPE, device=device)
    grid_xy = torch.tensor(grid_points(), dtype=DTYPE, device=device)

    model = build_deeponet({"n_sensors": n_sensors, "net": config["model"]}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    epochs = int(tr["epochs"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tr["lr_min"]))
    batch_size, n_query = int(tr["batch_size"]), int(tr["n_query"])
    eval_every, patience = int(tr["eval_every"]), int(tr["patience"])

    gen = torch.Generator().manual_seed(seed)
    n_train, n_pts = train_b.shape[0], grid_xy.shape[0]

    def val_error() -> float:
        """Mean relative L2 (degC fields, non-sensor nodes) on the validation split."""
        model.eval()
        with torch.no_grad():
            pred = model(val_b, grid_xy).cpu().numpy().reshape(-1, GRID, GRID) * scale + va_d["T_amb"][:, None, None]
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
        perm = torch.randperm(n_train, generator=gen)
        total, nb = 0.0, 0
        for s in range(0, n_train, batch_size):
            idx = perm[s:s + batch_size]
            q = torch.randint(0, n_pts, (n_query,), generator=gen)
            loss = torch.mean((model(train_b[idx], grid_xy[q]) - train_f[idx][:, q]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
            nb += 1
        sched.step()
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            v = val_error()
            if not quiet:
                print(f"[N={n_sensors} seed={seed}] epoch {epoch + 1}/{epochs} "
                      f"train_loss={total / nb:.5f} val_rel_l2={v:.5f}", flush=True)
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

    # ---- test: one row per held-out sample, non-sensor nodes only ---------- #
    model.eval()
    test_b = torch.tensor(branch_inputs(te_d, scale), dtype=DTYPE, device=device)
    rows = []
    with torch.no_grad():
        for i, sid in enumerate(te_d["ids"]):
            t1 = time.perf_counter()
            pred_norm = model.predict_field(test_b[i], grid_xy).cpu().numpy()
            inf_s = time.perf_counter() - t1
            pred = pred_norm.reshape(GRID, GRID) * scale + te_d["T_amb"][i]
            rows.append({"sample_id": sid, "n_sensors": n_sensors, "seed": seed,
                         **all_metrics(pred, te_d["T"][i], te_d["mask"][i]),
                         "pde_residual_rel": relative_pde_residual(pred, te_d["Q"][i], te_d["T_amb"][i]),
                         "inference_seconds": inf_s})
    maps = None
    if save_maps:
        with torch.no_grad():
            maps = []
            for i in range(min(save_maps, len(te_d["ids"]))):
                pf = model.predict_field(test_b[i], grid_xy).cpu().numpy()
                maps.append((te_d["ids"][i], pf.reshape(GRID, GRID) * scale + te_d["T_amb"][i],
                             te_d["T"][i], te_d["pos"][i]))
    return {"maps": maps, "rows": rows, "best_val_rel_l2": best["val"], "best_epoch": best["epoch"],
            "train_seconds": train_seconds, "model": model,
            "n_params": sum(p.numel() for p in model.parameters())}


def save_map_figure(maps: list, n_sensors: int, seed: int, out_dir: Path) -> Path:
    """True | predicted | abs error per test sample, sensors drawn as white dots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(maps), 3, figsize=(12, 3.6 * len(maps)), squeeze=False)
    for r, (sid, pred, true, pos) in enumerate(maps):
        vmin, vmax = true.min(), true.max()
        err = np.abs(pred - true)
        for c, (img, title, cmap, lim) in enumerate([
            (true, f"{sid}: true (C)", "inferno", (vmin, vmax)),
            (pred, "DeepONet prediction (C)", "inferno", (vmin, vmax)),
            (err, "absolute error (C)", "viridis", (0, None)),
        ]):
            im = axes[r, c].imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1], origin="upper")
            axes[r, c].scatter(pos[:, 0] * (GRID - 1), pos[:, 1] * (GRID - 1), c="white", s=18, edgecolors="k")
            axes[r, c].set_title(title)
            axes[r, c].axis("off")
            plt.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.suptitle(f"Model A, N={n_sensors} sensors, seed={seed}")
    fig.tight_layout()
    path = out_dir / f"maps_n{n_sensors}_seed{seed}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model A (DeepONet baseline, no physics)")
    ap.add_argument("--config", default="configs/deeponet.yaml")
    ap.add_argument("--sensors", default=None, help="comma list, overrides config n_sensors")
    ap.add_argument("--seeds", default=None, help="comma list, overrides config seeds")
    ap.add_argument("--strategy", default=None, help="overrides config sensor_strategy")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out-dir", default=str(RESULTS_DIR))
    ap.add_argument("--save-maps", type=int, default=0, help="save true/pred/error figures for first K test samples")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    sensors = [int(s) for s in args.sensors.split(",")] if args.sensors else config["n_sensors"]
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config["seeds"]
    strategy = args.strategy or config["sensor_strategy"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(RAW_DIR)
    splits = load_splits(RAW_DIR / "splits.csv")
    hw = "CPU" if config.get("device", "cpu") == "cpu" else torch.cuda.get_device_name(0)
    print(f"splits: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()) + f"; strategy={strategy}; hw={hw}")

    seed_rows = {s: [] for s in seeds}
    for n in sensors:
        data = {sp: load_split(splits[sp], n, strategy, meta) for sp in ("train", "validation", "test")}
        for seed in seeds:
            print(f"=== Model A: N={n}, seed={seed} ===", flush=True)
            res = train_one_config(n, seed, config, data, quiet=args.quiet, save_maps=args.save_maps)
            df = pd.DataFrame(res["rows"])
            df.to_csv(out_dir / f"per_sample_n{n}_seed{seed}.csv", index=False)
            summary = {"config": f"deeponet_{strategy}_n{n}", "seed": seed,
                       **df.drop(columns=["sample_id", "n_sensors", "seed"]).mean().to_dict(),
                       "best_val_rel_l2": res["best_val_rel_l2"], "best_epoch": res["best_epoch"],
                       "n_params": res["n_params"], "train_hours": res["train_seconds"] / 3600}
            seed_rows[seed].append(summary)
            # rewritten after every run so partial progress is never lost
            pd.DataFrame(seed_rows[seed]).to_csv(out_dir / f"seed_{seed}.csv", index=False)
            print(f"    best val rel_l2={res['best_val_rel_l2']:.4f} (epoch {res['best_epoch']}) "
                  f"-> test rel_l2={summary['relative_l2']:.4f}, MAE={summary['mae_C']:.2f} C, "
                  f"RMSE={summary['rmse_C']:.2f} C, params={res['n_params']}, "
                  f"train={res['train_seconds']:.0f}s", flush=True)
            if res["maps"]:
                print("    saved", save_map_figure(res["maps"], n, seed, out_dir))
            if args.save_checkpoints:
                ck = out_dir / "checkpoints"
                ck.mkdir(exist_ok=True)
                torch.save(res["model"].state_dict(), ck / f"deeponet_n{n}_seed{seed}.pt")

    from src.eval.report import aggregate_results
    md = aggregate_results(out_dir, out_dir / "summary.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
