"""Train and evaluate the sensor-conditioned FBPINN.

Uses the same data, targets, physics residual, metrics and model-selection rule as the FNO/PINO run,
so the two models can be compared directly.

Example:
    python -m src.train.fbpinn --sensors 5 --seed 0 --steps 3000
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data.dataset import load_metadata
from src.data.splits import load_splits
from src.models.fbpinn import FBPINN, grid_coordinates, sensor_features
from src.physics.grid_residual import TEMP_SCALE_C, grid_residual
from src.train.fno_pino import load_split_arrays, make_targets, physics_weight, score

DEFAULT_OUT = Path("results/fbpinn")


def _features(data: dict, idx: torch.Tensor) -> torch.Tensor:
    return sensor_features(data["positions"][idx], data["readings"][idx], data["T_amb"][idx])


@torch.no_grad()
def predict_fields(model: FBPINN, data: dict, coords: torch.Tensor, batch: int = 16) -> torch.Tensor:
    """Predicted temperature in degC for every chip, shape (n, 128, 128)."""
    model.eval()
    outputs = []
    for start in range(0, len(data["ids"]), batch):
        idx = torch.arange(start, min(start + batch, len(data["ids"])))
        theta, _ = model(_features(data, idx), coords)
        outputs.append(data["T_amb"][idx].view(-1, 1) + TEMP_SCALE_C * theta)
    model.train()
    return torch.cat(outputs).reshape(-1, 128, 128)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    splits = load_splits()
    meta = load_metadata()
    train_ids = splits["train"][: args.train_chips] if args.train_chips else splits["train"]
    print(f"loading {len(train_ids)} train / {len(splits['validation'])} val / {len(splits['test'])} test chips "
          f"with {args.sensors} sensors ...", flush=True)
    train_data = load_split_arrays(train_ids, args.sensors, meta)
    val_data = load_split_arrays(splits["validation"], args.sensors, meta)
    test_data = load_split_arrays(splits["test"], args.sensors, meta)
    theta_train, q_train = make_targets(train_data)
    coords = grid_coordinates()

    model = FBPINN(n_sensors=args.sensors, windows_per_side=args.windows, window_width=args.window_width,
                   embed_dim=args.embed, hidden=args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr_min)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}", flush=True)

    out_dir = args.out / f"N{args.sensors}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_val, best_state, best_step = [], float("inf"), None, 0
    start_time = time.time()

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_ids), (args.batch,))
        theta_pred, q_pred = model(_features(train_data, idx), coords)
        theta_pred = theta_pred.reshape(-1, 128, 128)
        q_pred = q_pred.reshape(-1, 128, 128)

        loss_temp = F.mse_loss(theta_pred, theta_train[idx])
        loss_src = F.mse_loss(q_pred, q_train[idx])
        residual = grid_residual(theta_pred, q_pred, train_data["T_amb"][idx])
        loss_phys = torch.mean(residual ** 2)
        w_phys = physics_weight(step, args.steps, args.w_phys, args.phys_warmup)
        loss = args.w_temp * loss_temp + args.w_src * loss_src + w_phys * loss_phys

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            val_metrics = score(predict_fields(model, val_data, coords), val_data)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "loss_temp": float(loss_temp.detach()),
                "loss_src": float(loss_src.detach()),
                "loss_phys": float(loss_phys.detach()),
                "val_relative_l2": val_metrics["relative_l2"],
                "elapsed_s": round(time.time() - start_time, 1),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if val_metrics["relative_l2"] < best_val:
                best_val = val_metrics["relative_l2"]
                best_state = copy.deepcopy(model.state_dict())
                best_step = step

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    final = {
        "model": "fbpinn",
        "sensors": args.sensors,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "parameters": n_params,
        "train_seconds": round(time.time() - start_time, 1),
        "validation": score(predict_fields(model, val_data, coords), val_data),
        "test": score(predict_fields(model, test_data, coords), test_data),
        "config": vars(args) | {"out": str(args.out)},
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print("test:", json.dumps(final["test"]), flush=True)
    return final


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensors", type=int, default=5, choices=[5, 8, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--w-temp", type=float, default=1.0)
    parser.add_argument("--w-src", type=float, default=0.1)
    parser.add_argument("--w-phys", type=float, default=0.01)
    parser.add_argument("--phys-warmup", type=float, default=0.2)
    parser.add_argument("--windows", type=int, default=3, help="windows per side (3 gives 9 windows)")
    parser.add_argument("--window-width", type=float, default=0.35)
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--train-chips", type=int, default=0, help="0 uses all 500 training chips")
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the torch default")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
