"""Model C, stage 2: per-chip physics refinement of the pretrained hybrid DeepONet.

Stage 1 (``train_hybrid.py`` with ``configs/hybrid_pretrain.yaml``) learns, from the
500 training chips, a fast map  sensors -> coefficients c (128) -> temperature
``theta(x) = c . phi(x) + b`` with ``phi`` a learned basis that has exactly
adiabatic edges (hard BC).

Stage 2 runs on every *new* chip and never touches the ground-truth field:

 1. ``c0``/``theta0`` = stage-1 prediction.
 2. Fit a few Gaussian power blobs (structured source, as in the physics-only
    PINN) to the source that ``theta0`` implies, so the heat equation starts
    close to satisfied.
 3. Optimise ``c`` and the blobs on
        w_data * |theta(sensors) - noisy readings|^2          (sensor fit)
      + w_pde  * mean(R^2)                                     (heat equation, FD on the 128 grid)
      + w_prior* mean((theta - theta0)^2)                      (stay near the stage-1 map)
    where ``R = [-t_die div(k grad T) + h_eff (T - T_amb) - Q_dyn * leak(T)] / (h_eff S)``.

Hyper-parameters are chosen on validation chips only (``--split validation``);
the test split is scored once with the chosen values.

Usage
-----
    python -m src.train.refine_hybrid --split validation --sensors 5 --seeds 0 --n-chips 20 --w-pde 1
    python -m src.train.refine_hybrid --split test --sensors 5,8,10 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import math
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
from src.physics.losses import Physics
from src.physics.residual import relative_pde_residual
from src.train.train_deeponet import branch_inputs, grid_points, load_split

DTYPE = torch.float32
CKPT_DIR = Path("results/model_c_pretrain")


# --------------------------------------------------------------------------- #
# structured source: sum of isotropic Gaussian blobs (after Model B's GaussianSource)
# --------------------------------------------------------------------------- #
class GaussianBlobs(torch.nn.Module):
    def __init__(self, n_blobs: int = 6, init_amp: float = 0.5, init_width: float = 0.12):
        super().__init__()
        g = int(math.ceil(math.sqrt(n_blobs)))
        axis = (torch.arange(g, dtype=DTYPE) + 0.5) / g
        gx, gy = torch.meshgrid(axis, axis, indexing="xy")
        pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)[:n_blobs].clamp(0.05, 0.95)
        self.centre_raw = torch.nn.Parameter(torch.log(pts / (1 - pts)))
        self.log_amp = torch.nn.Parameter(torch.full((n_blobs,), math.log(init_amp)))
        self.log_width = torch.nn.Parameter(torch.full((n_blobs,), math.log(init_width)))

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        d2 = ((xy[:, None, :] - torch.sigmoid(self.centre_raw)[None]) ** 2).sum(-1)
        w = torch.exp(self.log_width).clamp(0.02, 0.5)
        return (torch.exp(self.log_amp)[None] * torch.exp(-d2 / (2 * w[None] ** 2))).sum(1)


# --------------------------------------------------------------------------- #
# finite-difference residual of a 128x128 field, torch / differentiable
# --------------------------------------------------------------------------- #
def fd_residual(T_C: torch.Tensor, T_amb: float, physics: Physics, Q_total: torch.Tensor | None = None):
    """Returns interior ``-t_die div(k grad T) + h_eff (T - T_amb) [- Q_total]`` in W/m^2."""
    n = T_C.shape[0]
    d = physics.L / (n - 1)
    k = physics.k(T_C)
    fx = 0.5 * (k[:, 1:] + k[:, :-1]) * (T_C[:, 1:] - T_C[:, :-1]) / d
    fy = 0.5 * (k[1:, :] + k[:-1, :]) * (T_C[1:, :] - T_C[:-1, :]) / d
    div = (fx[1:-1, 1:] - fx[1:-1, :-1]) / d + (fy[1:, 1:-1] - fy[:-1, 1:-1]) / d
    out = -physics.t_die * div + physics.h_eff * (T_C[1:-1, 1:-1] - T_amb)
    return out if Q_total is None else out - Q_total[1:-1, 1:-1]


def load_model(n_sensors: int, seed: int, config: dict, ckpt_dir: Path = CKPT_DIR):
    model = build_hybrid({"n_sensors": n_sensors, "net": config["model"]})
    path = ckpt_dir / f"n{n_sensors}" / "checkpoints" / f"{config['name']}_n{n_sensors}_seed{seed}.pt"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()


def refine_chip(model, b_in: torch.Tensor, Phi: torch.Tensor, grid_xy: torch.Tensor, sensor_flat: np.ndarray,
                y_sensor: torch.Tensor, T_amb: float, scale: float, physics: Physics, hp: dict):
    """Refine one chip. Returns ``(theta0, theta)`` as ``(GRID*GRID,)`` tensors (normalised)."""
    p = model.p
    with torch.no_grad():
        c0 = model.embed_branch(b_in[None])[0, :p].clone()
        theta0 = Phi @ c0 + model.bias_T
    hS = physics.h_eff * scale

    def to_grid_C(theta):
        return (T_amb + scale * theta).reshape(GRID, GRID)

    # --- initialise the source blobs on the source theta0 implies ----------- #
    with torch.no_grad():
        T0 = to_grid_C(theta0)
        q_imp = (fd_residual(T0, T_amb, physics) / hS).clamp_min(0)
        q_dyn_target = F_pad(q_imp) / physics.leakage_factor(T0)
    blobs = GaussianBlobs(int(hp["n_blobs"]))
    opt_b = torch.optim.Adam(blobs.parameters(), lr=0.05)
    for _ in range(int(hp["blob_init_steps"])):
        loss = torch.mean((blobs(grid_xy).reshape(GRID, GRID) - q_dyn_target) ** 2)
        opt_b.zero_grad()
        loss.backward()
        opt_b.step()

    # --- joint refinement ---------------------------------------------------- #
    c = torch.nn.Parameter(c0.clone())
    opt = torch.optim.Adam([{"params": [c], "lr": float(hp["lr_c"])},
                            {"params": blobs.parameters(), "lr": float(hp["lr_blobs"])}])
    for _ in range(int(hp["steps"])):
        theta = Phi @ c + model.bias_T
        T_C = to_grid_C(theta)
        q_tot = F_pad_full(blobs(grid_xy).reshape(GRID, GRID) * physics.leakage_factor(T_C) * hS)
        res = fd_residual(T_C, T_amb, physics, q_tot) / hS
        loss = (float(hp["w_data"]) * torch.mean((theta[sensor_flat] - y_sensor) ** 2)
                + float(hp["w_pde"]) * torch.mean(res ** 2)
                + float(hp["w_prior"]) * torch.mean((theta - theta0) ** 2))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return theta0, Phi @ c + model.bias_T


def F_pad(x: torch.Tensor) -> torch.Tensor:
    """Interior ``(n-2, n-2)`` -> full ``(n, n)`` by edge replication."""
    return torch.nn.functional.pad(x[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]


def F_pad_full(x: torch.Tensor) -> torch.Tensor:
    return x


def refine_split(split: str, n_sensors: int, seed: int, config: dict, hp: dict, data: dict,
                 n_chips: int | None = None) -> list[dict]:
    physics = Physics.from_config(config.get("physics"))
    model = load_model(n_sensors, seed, config)
    tr = data["train"]
    scale = float((tr["T"] - tr["T_amb"][:, None, None]).max())
    d = data[split]
    grid_xy = torch.tensor(grid_points(), dtype=DTYPE)
    with torch.no_grad():
        Phi = model.trunk_T(model._trunk_T_in(grid_xy))
    b_all = torch.tensor(branch_inputs(d, scale), dtype=DTYPE)
    ids = list(range(len(d["ids"]))) if n_chips is None else list(range(n_chips))
    rows = []
    for i in ids:
        pos = d["pos"][i]
        flat = np.round(pos[:, 1] * (GRID - 1)).astype(int) * GRID + np.round(pos[:, 0] * (GRID - 1)).astype(int)
        y_s = torch.tensor((d["noisy"][i] - d["T_amb"][i]) / scale, dtype=DTYPE)
        t1 = time.perf_counter()
        th0, th = refine_chip(model, b_all[i], Phi, grid_xy, flat, y_s, float(d["T_amb"][i]), scale, physics, hp)
        sec = time.perf_counter() - t1
        for stage, theta in (("stage1", th0), ("refined", th)):
            pred = theta.numpy().reshape(GRID, GRID).astype(np.float64) * scale + d["T_amb"][i]
            rows.append({"stage": stage, "sample_id": d["ids"][i], "n_sensors": n_sensors, "seed": seed,
                         **all_metrics(pred, d["T"][i], d["mask"][i]),
                         "pde_residual_rel": relative_pde_residual(pred, d["Q"][i], d["T_amb"][i]),
                         "refine_seconds": sec if stage == "refined" else 0.0})
    return rows


DEFAULT_HP = {"n_blobs": 6, "blob_init_steps": 300, "steps": 300, "lr_c": 1e-3, "lr_blobs": 1e-2,
              "w_data": 1.0, "w_pde": 1.0, "w_prior": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description="Model C stage 2: per-chip physics refinement")
    ap.add_argument("--config", default="configs/hybrid_pretrain.yaml")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--sensors", default="5")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--n-chips", type=int, default=None)
    ap.add_argument("--out", default=None, help="CSV of per-chip rows")
    for k, v in DEFAULT_HP.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    args = ap.parse_args()
    hp = {k: getattr(args, k) for k in DEFAULT_HP}
    config = yaml.safe_load(open(args.config))
    meta, splits = load_metadata(RAW_DIR), load_splits(RAW_DIR / "splits.csv")

    all_rows = []
    for n in [int(s) for s in args.sensors.split(",")]:
        data = {sp: load_split(splits[sp], n, config["sensor_strategy"], meta) for sp in ("train", args.split)}
        for seed in [int(s) for s in args.seeds.split(",")]:
            rows = refine_split(args.split, n, seed, config, hp, data, args.n_chips)
            df = pd.DataFrame(rows)
            g = df.groupby("stage")[["relative_l2", "mae_C", "rmse_C", "pde_residual_rel"]].mean()
            print(f"N={n} seed={seed} [{args.split}] hp={ {k: hp[k] for k in ('w_data','w_pde','w_prior','steps','lr_c')} }")
            print(g.round(4).to_string(), flush=True)
            all_rows += rows
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
