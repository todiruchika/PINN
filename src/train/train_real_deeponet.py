"""DeepONet on the real commercial-processor thermal maps, one model per (processor, sensor count, seed).

Same recipe as Model A on the simulated chips: the branch net sees the sensor positions and noisy readings, the trunk
net a query coordinate, output = temperature there. Differences forced by the data:

* no ambient temperature is known, so maps are standardised with the training-set mean and std (all pixels, all frames);
* map size differs per processor, coordinates are in the unit square (x = col/(W-1), y = row/(H-1));
* sensor readings are the map value at the pixel plus the same sensor model as before (Gaussian 1 degC, rounded to
  1 degC, fixed per-sensor offset of std 0.5 degC), because these maps are already camera measurements;
* sensors are placed by POD-QR on the training maps of that processor only;
* splits keep near-duplicate frames apart (see ``src/data/real_thermal.py``).

Two references are scored on the same test frames and the same sensor readings: the **training-mean map** (what you get
with no sensors) and **gappy POD** (a classical, non-neural reconstruction from the same POD basis).

Metrics are on non-sensor pixels. Besides the protocol's relative L2 on temperature, ``rel_l2_rise`` is relative to each
frame's rise above its own minimum, which is the fairer scale here: absolute temperatures sit near 60 to 90 degC, so the
plain relative L2 looks small even for a poor reconstruction.

Usage
-----
    python -m src.data.real_thermal --download
    python -m src.train.train_real_deeponet --processor i7 --sensors 5 --seeds 0 --steps 500     # smoke test
    python -m src.train.train_real_deeponet --processor i7
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.real_thermal import PROCESSORS, load_workloads, split_frames, stack
from src.eval.metrics import all_metrics
from src.models.deeponet import build_deeponet
from src.sensors.real_podqr import pod_basis, podqr_sensors

DTYPE = torch.float32
RESULTS = Path("results/real_deeponet")


def grid_points(H: int, W: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


def read_sensors(maps: np.ndarray, rc: np.ndarray, first_frame: int, n: int, cfg: dict, split_id: int) -> np.ndarray:
    """Noisy readings ``(frames, n)`` at the first ``n`` sensor pixels."""
    s = cfg["sensors"]
    true = maps[:, rc[:n, 0], rc[:n, 1]].astype(np.float64)
    offsets = np.random.default_rng(42 + 1000).normal(0, s["calib_sigma_C"], size=10)[:n]
    out = np.empty_like(true)
    for k in range(len(true)):
        rng = np.random.default_rng(42 * 10000 + split_id * 100000 + (first_frame + k) * 100 + n)
        out[k] = np.round((true[k] + offsets + rng.normal(0, s["noise_sigma_C"], size=n)) / s["quant_step_C"]) * s["quant_step_C"]
    return out


def branch_inputs(readings: np.ndarray, rc: np.ndarray, shape: tuple, mu: float, sd: float, n: int) -> np.ndarray:
    H, W = shape
    pos = np.stack([rc[:n, 1] / (W - 1), rc[:n, 0] / (H - 1)], axis=1)
    order = np.lexsort((pos[:, 1], pos[:, 0]))
    theta = (readings - mu) / sd
    out = np.zeros((len(readings), 3 * n), dtype=np.float32)
    for i in range(len(readings)):
        out[i] = np.concatenate([pos[order], theta[i][order, None]], axis=1).ravel()
    return out


def metrics_row(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> dict:
    m = all_metrics(pred, true, mask)
    rise = true - true.min()
    m["rel_l2_rise"] = float(np.linalg.norm((pred - true)[mask]) / (np.linalg.norm(rise[mask]) + 1e-12))
    return m


def prepare(processor: str, cfg: dict) -> dict:
    wl = load_workloads(processor)
    sp = split_frames(processor, wl, seed=cfg["split_seed"])
    maps = {k: stack(wl, v) for k, v in sp.items()}
    shape = maps["train"].shape[1:]
    mu, sd = float(maps["train"].mean()), float(maps["train"].std())
    mean, basis = pod_basis((maps["train"] - mu) / sd, cfg["sensors"]["energy"], max(cfg["sensor_counts"]))
    rc = podqr_sensors(basis, shape, max(cfg["sensor_counts"]), cfg["sensors"]["margin_px"])
    return {"maps": maps, "shape": shape, "mu": mu, "sd": sd, "pod_mean": mean, "pod_basis": basis, "rc": rc, "frames": sp}


def baselines(D: dict, n: int, cfg: dict, readings: dict, return_preds: bool = False):
    """Training-mean map and gappy POD, same sensor readings as the DeepONet. One row per test frame each.

    Gappy POD fits ``k`` POD coefficients to the ``n`` noisy readings by least squares. ``k`` (1..n) is chosen on the
    validation frames, so the reference is not handicapped by an ill-conditioned fit.
    """
    H, W = D["shape"]
    mask = np.ones((H, W), dtype=bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
    mean_map = D["maps"]["train"].mean(axis=0)
    flat = D["rc"][:n, 0] * W + D["rc"][:n, 1]

    def gappy(read, k):
        Psi = D["pod_basis"][:, :k]
        out = []
        for y in read:
            coef, *_ = np.linalg.lstsq(Psi[flat], (y - D["mu"]) / D["sd"] - D["pod_mean"][flat], rcond=None)
            out.append(((D["pod_mean"] + Psi @ coef) * D["sd"] + D["mu"]).reshape(H, W))
        return out

    def val_err(k):
        p = gappy(readings["validation"], k)
        return float(np.mean([np.linalg.norm((p[i] - t)[mask]) / np.linalg.norm(t[mask]) for i, t in enumerate(D["maps"]["validation"])]))

    k_best = min(range(1, n + 1), key=val_err)
    preds = gappy(readings["test"], k_best)
    rows = []
    for k, true in enumerate(D["maps"]["test"]):
        rows.append({"model": "train-mean map", "frame": k, **metrics_row(mean_map, true, mask)})
        rows.append({"model": f"gappy POD", "frame": k, "pod_modes": k_best, **metrics_row(preds[k], true, mask)})
    return (rows, preds) if return_preds else rows


def train_one(D: dict, n: int, seed: int, cfg: dict, readings: dict, quiet: bool = True, save_maps: int = 0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = cfg["training"]
    H, W = D["shape"]
    mu, sd, rc = D["mu"], D["sd"], D["rc"]
    b = {k: torch.tensor(branch_inputs(readings[k], rc, D["shape"], mu, sd, n), dtype=DTYPE) for k in readings}
    f_train = torch.tensor(((D["maps"]["train"] - mu) / sd).reshape(len(D["maps"]["train"]), -1), dtype=DTYPE)
    xy = torch.tensor(grid_points(H, W), dtype=DTYPE)
    model = build_deeponet({"n_sensors": n, "net": cfg["model"]})
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    steps = int(tr["steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=float(tr["lr_min"]))
    gen = torch.Generator().manual_seed(seed)
    n_train, n_pts = f_train.shape
    mask_val = np.ones((H, W), dtype=bool); mask_val[rc[:n, 0], rc[:n, 1]] = False

    def predict(split: str) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return (model(b[split], xy).numpy().reshape(-1, H, W) * sd + mu)

    def val_error() -> float:
        p, t = predict("validation"), D["maps"]["validation"]
        return float(np.mean([np.linalg.norm((p[i] - t[i])[mask_val]) / np.linalg.norm(t[i][mask_val]) for i in range(len(t))]))

    best, bad, t0 = {"v": float("inf"), "state": None, "step": 0}, 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, n_train, (int(tr["batch_size"]),), generator=gen)
        q = torch.randint(0, n_pts, (int(tr["n_query"]),), generator=gen)
        loss = torch.mean((model(b["train"][idx], xy[q]) - f_train[idx][:, q]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % int(tr["eval_every"]) == 0 or step == steps:
            v = val_error()
            if not quiet:
                print(f"  [{n} sensors, seed {seed}] step {step}/{steps} loss={float(loss.detach()):.5f} val_rel_l2={v:.5f}", flush=True)
            if v < best["v"]:
                best = {"v": v, "step": step, "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}; bad = 0
            else:
                bad += 1
                if bad >= int(tr["patience"]):
                    break
    train_s = time.perf_counter() - t0
    model.load_state_dict(best["state"])
    pred = predict("test")
    rows = [{"model": "DeepONet", "frame": k, **metrics_row(pred[k], D["maps"]["test"][k], mask_val)} for k in range(len(pred))]
    maps = [(k, pred[k], D["maps"]["test"][k]) for k in range(min(save_maps, len(pred)))]
    return rows, best, train_s, sum(p.numel() for p in model.parameters()), maps


def save_map_figure(maps, rc, n, shape, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(maps), 3, figsize=(10, 3.2 * len(maps)), squeeze=False)
    for r, (k, pred, true) in enumerate(maps):
        for c, (img, t, cm, lim) in enumerate([(true, f"true (frame {k})", "inferno", (true.min(), true.max())),
                                                (pred, "DeepONet", "inferno", (true.min(), true.max())),
                                                (np.abs(pred - true), "absolute error", "viridis", (0, None))]):
            im = axes[r, c].imshow(img, cmap=cm, vmin=lim[0], vmax=lim[1])
            axes[r, c].scatter(rc[:n, 1], rc[:n, 0], c="white", s=16, edgecolors="k")
            axes[r, c].set_title(t); axes[r, c].axis("off"); plt.colorbar(im, ax=axes[r, c], fraction=0.046)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepONet on real thermal maps")
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--processor", default="i7", choices=list(PROCESSORS))
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-maps", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps:
        cfg["training"]["steps"] = a.steps
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    out = Path(a.out_dir or RESULTS / a.processor); out.mkdir(parents=True, exist_ok=True)

    D = prepare(a.processor, cfg)
    print(f"{a.processor}: map {D['shape']}, split " + ", ".join(f"{k}={len(v)}" for k, v in D["maps"].items())
          + f"; POD-QR sensors (row,col) = {D['rc'].tolist()}", flush=True)
    summary = []
    for n in sensors:
        readings = {}
        for sid, k in enumerate(("train", "validation", "test")):
            first = D["frames"][k][0][1] if k != "train" and len(set(w for w, _ in D["frames"][k])) == 1 else 0
            readings[k] = read_sensors(D["maps"][k], D["rc"], first, n, cfg, sid)
        base = pd.DataFrame(baselines(D, n, cfg, readings))
        for model_name, g in base.groupby("model"):
            summary.append({"processor": a.processor, "n_sensors": n, "model": model_name, "seed": -1,
                            **g.drop(columns=["model", "frame"]).mean().to_dict()})
        for seed in seeds:
            print(f"=== {a.processor}: N={n}, seed={seed} ===", flush=True)
            rows, best, secs, n_params, maps = train_one(D, n, seed, cfg, readings, a.quiet, a.save_maps)
            df = pd.DataFrame(rows); df.to_csv(out / f"per_frame_n{n}_seed{seed}.csv", index=False)
            summary.append({"processor": a.processor, "n_sensors": n, "model": "DeepONet", "seed": seed,
                            **df.drop(columns=["model", "frame"]).mean().to_dict(), "best_val_rel_l2": best["v"],
                            "best_step": best["step"], "n_params": n_params, "train_seconds": secs})
            if maps:
                save_map_figure(maps, D["rc"], n, D["shape"], out / f"maps_n{n}_seed{seed}.png", f"{a.processor}, {n} sensors, seed {seed}")
            m = df.mean(numeric_only=True)
            print(f"    test rel_l2={m.relative_l2:.4f}  rel_l2_rise={m.rel_l2_rise:.3f}  MAE={m.mae_C:.2f} C  RMSE={m.rmse_C:.2f} C  "
                  f"(val {best['v']:.4f}, best step {best['step']}, {secs:.0f}s)", flush=True)
            pd.DataFrame(summary).to_csv(out / "summary_runs.csv", index=False)
    S = pd.DataFrame(summary)
    S.to_csv(out / "summary_runs.csv", index=False)
    cols = ["relative_l2", "rel_l2_rise", "mae_C", "rmse_C", "peak_temp_error_C", "hotspot_localization_cells"]
    agg = S.groupby(["n_sensors", "model"])[cols].agg(["mean", "std"])
    lines = ["| sensors | model | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for (n, mname), row in agg.iterrows():
        lines.append(f"| {n} | {mname} | " + " | ".join(f"{row[(c, 'mean')]:.4f}" + (f" ± {row[(c, 'std')]:.4f}" if row[(c, 'std')] == row[(c, 'std')] else "") for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
