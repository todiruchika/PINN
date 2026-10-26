"""Train the sensor-to-source network, with (hybrid) or without (ml) the built-in heat-equation solver.

Example:
    python -m src.train.phys_hybrid --mode hybrid --sensors 10 --seed 0 --steps 6000
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
from src.models.fbpinn import sensor_features
from src.models.phys_decoder import SensorToField
from src.physics.grid_residual import TEMP_SCALE_C
from src.train.fno_pino import load_split_arrays, make_targets, score


def _features(data: dict, idx: torch.Tensor) -> torch.Tensor:
    return sensor_features(data["positions"][idx], data["readings"][idx], data["T_amb"][idx])


@torch.no_grad()
def predict_fields(model: SensorToField, data: dict, batch: int = 50) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(data["ids"]), batch):
        idx = torch.arange(start, min(start + batch, len(data["ids"])))
        theta, _ = model(_features(data, idx), data["T_amb"][idx])
        outputs.append(data["T_amb"][idx].view(-1, 1, 1) + TEMP_SCALE_C * theta)
    model.train()
    return torch.cat(outputs)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    splits, meta = load_splits(), load_metadata()
    train_ids = splits["train"][: args.train_chips] if args.train_chips else splits["train"]
    train_data = load_split_arrays(train_ids, args.sensors, meta)
    val_data = load_split_arrays(splits["validation"], args.sensors, meta)
    test_data = load_split_arrays(splits["test"], args.sensors, meta)
    theta_train, q_train = make_targets(train_data)

    model = SensorToField(args.sensors, args.mode, args.encoder_width, args.channels, dropout=args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"mode {args.mode}, parameters {n_params:,}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr_min)
    out_dir = args.out / f"N{args.sensors}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_val, best_state, best_step = [], float("inf"), None, 0
    start = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(train_ids), (args.batch,))
        features = _features(train_data, idx)
        if args.noise_c > 0:   # fresh sensor noise each step: readings are in units of 50 K, in feature slots 2, 5, ...
            noise = torch.randn(features.shape[0], args.sensors) * args.noise_c / TEMP_SCALE_C
            features = features.clone()
            features[:, 2:3 * args.sensors:3] += noise
        theta_pred, q_pred = model(features, train_data["T_amb"][idx])
        loss_temp = F.mse_loss(theta_pred, theta_train[idx])
        loss_src = F.mse_loss(q_pred, q_train[idx])
        loss = loss_temp + args.w_src * loss_src
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == args.steps:
            val = score(predict_fields(model, val_data), val_data)
            record = {"step": step, "loss_temp": float(loss_temp), "loss_src": float(loss_src),
                      "val_relative_l2": val["relative_l2"], "elapsed_s": round(time.time() - start, 1)}
            history.append(record)
            print(json.dumps(record), flush=True)
            if val["relative_l2"] < best_val:
                best_val, best_step, best_state = val["relative_l2"], step, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    final = {"model": f"phys_decoder_{args.mode}", "sensors": args.sensors, "seed": args.seed, "steps": args.steps,
             "batch": args.batch, "best_step": best_step, "parameters": n_params,
             "train_seconds": round(time.time() - start, 1),
             "validation": score(predict_fields(model, val_data), val_data),
             "test": score(predict_fields(model, test_data), test_data),
             "config": vars(args) | {"out": str(args.out)}, "history": history}
    (out_dir / "results.json").write_text(json.dumps(final, indent=2, default=str), encoding="utf-8")
    print("test:", json.dumps(final["test"]), flush=True)
    return final


def parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["ml", "hybrid"], default="hybrid")
    p.add_argument("--sensors", type=int, default=10, choices=[5, 8, 10])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--w-src", type=float, default=0.1)
    p.add_argument("--encoder-width", type=int, default=256)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--noise-c", type=float, default=0.0, help="extra Gaussian sensor noise (deg C) added each training step")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--train-chips", type=int, default=0)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/hybrid"))
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
