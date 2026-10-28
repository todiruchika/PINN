"""FNO on the real commercial-processor thermal maps, trained exactly like the DeepONet of ``train_real_deeponet``.

Same data, splits, sensors and noisy readings (``prepare``, ``read_sensors``), same normalisation (training mean and std), same
recipe (``configs/real_deeponet.yaml``: 6000 steps, batch 32, Adam 1e-3 with cosine decay to 1e-5, validation every 200 steps,
patience 10, best checkpoint on the validation relative L2). The FNO predicts the whole map at once, so the loss is the MSE over
all pixels of the batch instead of 1024 sampled query points. Architecture is fixed (no search): width 16, 6 Fourier modes,
3 layers, about 114k parameters (the DeepONet has about 118k). Input encoding: see ``src/models/real_fno.py``.

Stage ``train`` trains the FNO (``--model fno``) or retrains the DeepONet (``--model deeponet``; ``train_one`` unchanged) and saves
their predictions on the sampled validation and test frames (``subset_idx``, the frames Ruchika's PINN was solved on) for the
hybrids. The prediction files are git-ignored.

    python -m src.train.train_real_fno --model fno --processor i7 --seeds 0 --sensors 5 --steps 500     # smoke test
    python -m src.train.train_real_fno --model fno --processor i7
    python -m src.train.train_real_fno --model deeponet --processor i7 --k-workloads 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.real_hybrid_pinn import restrict_training
from src.eval.real_pinn_runs import SUBSET, subset_idx
from src.models.real_fno import RealFNO, grid_inputs
from src.train.train_real_deeponet import metrics_row, prepare, read_sensors, train_one

DTYPE = torch.float32
PREDS = Path("results/real_new_models/preds")       # git-ignored (results/real_new_models/.gitignore)
SPLITS = ("train", "validation", "test")


def split_readings(D: dict, n: int, cfg: dict) -> dict:
    """The DeepONet's noisy readings for every split (same call as ``train_real_deeponet.main``)."""
    out = {}
    for sid, k in enumerate(SPLITS):
        first = D["frames"][k][0][1] if k != "train" and len({w for w, _ in D["frames"][k]}) == 1 else 0
        out[k] = read_sensors(D["maps"][k], D["rc"], first, n, cfg, sid)
    return out


def train_fno(D: dict, n: int, seed: int, cfg: dict, readings: dict, quiet: bool = True):
    """Returns ``(best, train_seconds, n_params, {"validation": maps, "test": maps})`` in degC."""
    torch.manual_seed(seed); np.random.seed(seed)
    tr = cfg["training"]
    H, W = D["shape"]
    mu, sd, rc = D["mu"], D["sd"], D["rc"]
    x = {k: torch.tensor(grid_inputs(readings[k], rc, D["shape"], mu, sd, n)) for k in readings}
    f_train = torch.tensor((D["maps"]["train"] - mu) / sd, dtype=DTYPE)
    model = RealFNO()
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    steps = int(tr["steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=float(tr["lr_min"]))
    gen = torch.Generator().manual_seed(seed)
    mask = np.ones((H, W), dtype=bool); mask[rc[:n, 0], rc[:n, 1]] = False

    def predict(split: str) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(x[split][i:i + 64]).numpy() for i in range(0, len(x[split]), 64)]) * sd + mu

    def val_error() -> float:
        p, t = predict("validation"), D["maps"]["validation"]
        return float(np.mean([np.linalg.norm((p[i] - t[i])[mask]) / np.linalg.norm(t[i][mask]) for i in range(len(t))]))

    best, bad, t0 = {"v": float("inf"), "state": None, "step": 0}, 0, time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        idx = torch.randint(0, len(f_train), (int(tr["batch_size"]),), generator=gen)
        loss = torch.mean((model(x["train"][idx]) - f_train[idx]) ** 2)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % int(tr["eval_every"]) == 0 or step == steps:
            v = val_error()
            if not quiet:
                print(f"  [{n} sensors, seed {seed}] step {step}/{steps} loss={float(loss.detach()):.5f} val_rel_l2={v:.5f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if v < best["v"]:
                best = {"v": v, "step": step, "state": {k: t.detach().clone() for k, t in model.state_dict().items()}}; bad = 0
            else:
                bad += 1
                if bad >= int(tr["patience"]):
                    break
    secs = time.perf_counter() - t0
    model.load_state_dict(best["state"])
    return best, secs, sum(p.numel() for p in model.parameters()), {"validation": predict("validation"), "test": predict("test")}


def run(model_name: str, proc: str, cfg: dict, sensors: list, seeds: list, k_workloads: int | None, quiet: bool) -> None:
    D = prepare(proc, cfg)
    tag = proc
    if k_workloads:
        chosen = restrict_training(D, k_workloads)
        tag = f"{proc}_low{k_workloads}"
        print(f"{tag}: training workloads = {chosen} ({len(D['maps']['train'])} frames)", flush=True)
    out = PREDS / model_name / tag; out.mkdir(parents=True, exist_ok=True)
    idx = {s: subset_idx(len(D["maps"][s]), SUBSET[proc]) for s in ("validation", "test")}
    log = out / "runs.csv"
    rows = pd.read_csv(log).to_dict("records") if log.exists() else []
    for n in sensors:
        readings = split_readings(D, n, cfg)
        H, W = D["shape"]
        mask = np.ones((H, W), dtype=bool); mask[D["rc"][:n, 0], D["rc"][:n, 1]] = False
        for seed in seeds:
            f = out / f"{model_name}_n{n}_seed{seed}.npz"
            if f.exists():
                continue
            if model_name == "fno":
                best, secs, n_params, preds = train_fno(D, n, seed, cfg, readings, quiet)
            else:
                _, best, secs, n_params, _, preds = train_one(D, n, seed, cfg, readings, quiet=quiet, return_preds=True)
            np.savez_compressed(f, val_idx=idx["validation"], test_idx=idx["test"],
                                val_pred=preds["validation"][idx["validation"]].astype(np.float32),
                                test_pred=preds["test"][idx["test"]].astype(np.float32))
            full = pd.DataFrame([metrics_row(preds["test"][k], t, mask) for k, t in enumerate(D["maps"]["test"])]).mean()
            rows.append({"processor": proc, "experiment": tag, "n_sensors": n, "seed": seed, "model": model_name,
                         "n_params": n_params, "train_seconds": secs, "best_step": best["step"], "best_val_rel_l2": best["v"],
                         "all_test_frames_rel_l2_rise": full.rel_l2_rise})
            pd.DataFrame(rows).to_csv(log, index=False)
            print(f"{tag} {model_name} N={n} seed={seed}: {secs:.0f}s, best step {best['step']}, val {best['v']:.4f}, "
                  f"test rise error (all {len(D['maps']['test'])} frames) {full.rel_l2_rise:.3f}, {n_params} params", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="FNO (and DeepONet retrain) on real thermal maps")
    ap.add_argument("--model", required=True, choices=["fno", "deeponet"])
    ap.add_argument("--processor", required=True, choices=["i5", "i7"])
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--steps", type=int, default=None, help="override for smoke tests only")
    ap.add_argument("--k-workloads", type=int, default=None, help="few-training-workloads experiment (i7)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.steps:
        cfg["training"]["steps"] = a.steps
        global PREDS
        PREDS = PREDS.parent / "smoke"
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    run(a.model, a.processor, cfg, sensors, seeds, a.k_workloads, a.quiet)


if __name__ == "__main__":
    main()
