"""Train and evaluate the FNO/PINO thermal-map model.

One network is trained on the training chips, with the validation chips used to
pick the best checkpoint. The test chips are scored once at the end, on the
non-sensor nodes only (PROTOCOL.md, Evaluation Grid).

Example:
    python -m src.train.fno_pino --sensors 10 --seed 0 --steps 3000
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

from generator.pde_solver import H_EFF_DEFAULT
from src.data.dataset import load_metadata, load_sample
from src.data.splits import load_splits
from src.eval.metrics import all_metrics
from src.models.fno_pino import FNOPINO, build_grid_input
from src.physics.grid_residual import TEMP_SCALE_C, grid_residual
from src.sensors.readers import sensor_mask

DEFAULT_OUT = Path("results/fno_pino")


def load_split_arrays(split_ids: list, n_sensors: int, meta) -> dict:
    """Read every chip in one split into memory."""
    T, Q, T_amb, positions, readings = [], [], [], [], []
    for sample_id in split_ids:
        sample = load_sample(sample_id, n_sensors, meta=meta)
        T.append(sample["T"])
        Q.append(sample["Q"])
        T_amb.append(float(sample["meta"]["T_amb_C"]))
        positions.append(np.asarray(sample["sensors"]["positions"], dtype=np.float32))
        readings.append(np.asarray(sample["sensors"]["temperatures_noisy"], dtype=np.float32))
    return {
        "ids": split_ids,
        "T": torch.tensor(np.stack(T), dtype=torch.float32),
        "Q": torch.tensor(np.stack(Q), dtype=torch.float32),
        "T_amb": torch.tensor(T_amb, dtype=torch.float32),
        "positions": torch.tensor(np.stack(positions)),
        "readings": torch.tensor(np.stack(readings)),
    }


def make_targets(data: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """theta = (T - T_amb) / S and q_tilde = Q / (h_eff * S)."""
    theta = (data["T"] - data["T_amb"].view(-1, 1, 1)) / TEMP_SCALE_C
    q_tilde = data["Q"] / (H_EFF_DEFAULT * TEMP_SCALE_C)
    return theta, q_tilde


def batch_input(data: dict, idx: torch.Tensor) -> torch.Tensor:
    return build_grid_input(data["positions"][idx], data["readings"][idx], data["T_amb"][idx])


def physics_weight(step: int, total: int, target: float, warmup_fraction: float) -> float:
    """Ramp the physics weight up from 0 over the first part of training."""
    warmup_steps = max(1, int(warmup_fraction * total))
    return target * min(1.0, step / warmup_steps)


@torch.no_grad()
def predict_fields(model: FNOPINO, data: dict, batch: int = 32) -> torch.Tensor:
    """Predicted temperature in degC for every chip, shape (n, 128, 128)."""
    model.eval()
    outputs = []
    for start in range(0, len(data["ids"]), batch):
        idx = torch.arange(start, min(start + batch, len(data["ids"])))
        theta, _ = model(batch_input(data, idx))
        outputs.append(data["T_amb"][idx].view(-1, 1, 1) + TEMP_SCALE_C * theta)
    model.train()
    return torch.cat(outputs)


def score(pred_T: torch.Tensor, data: dict) -> dict:
    """Mean metrics over chips, on non-sensor nodes only."""
    rows = []
    for i in range(len(data["ids"])):
        mask = ~sensor_mask(data["positions"][i].tolist())
        rows.append(all_metrics(pred_T[i].numpy(), data["T"][i].numpy(), mask))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    splits = load_splits()
    meta = load_metadata()
    train_ids = splits["train"][: args.train_chips] if args.train_chips else splits["train"]
    val_ids = splits["validation"]
    test_ids = splits["test"]

    print(f"loading {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test chips "
          f"with {args.sensors} sensors ...", flush=True)
    train_data = load_split_arrays(train_ids, args.sensors, meta)
    val_data = load_split_arrays(val_ids, args.sensors, meta)
    test_data = load_split_arrays(test_ids, args.sensors, meta)
    theta_train, q_train = make_targets(train_data)

    model = FNOPINO(width=args.width, modes=args.modes, depth=args.depth)
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
        grid_in = batch_input(train_data, idx)
        theta_pred, q_pred = model(grid_in)

        if args.hot_weight > 0:
            # Weight each pixel by how hot it is relative to its chip's peak, so
            # under-predicting a hotspot costs more than under-predicting background.
            target = theta_train[idx]
            rel = target / target.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
            weight = 1.0 + args.hot_weight * rel.clamp(min=0)
            loss_temp = (weight * (theta_pred - target) ** 2).sum() / weight.sum()
        else:
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
            val_metrics = score(predict_fields(model, val_data), val_data)
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
        "model": "fno_pino",
        "sensors": args.sensors,
        "seed": args.seed,
        "steps": args.steps,
        "best_step": best_step,
        "parameters": n_params,
        "train_seconds": round(time.time() - start_time, 1),
        "validation": score(predict_fields(model, val_data), val_data),
        "test": score(predict_fields(model, test_data), test_data),
        "config": vars(args) | {"out": str(args.out)},
        "history": history,
    }
    (out_dir / "results.json").write_text(json.dumps(final, indent=2, default=str))
    print("test:", json.dumps(final["test"]), flush=True)
    return final


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensors", type=int, default=10, choices=[5, 8, 10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--w-temp", type=float, default=1.0)
    parser.add_argument("--w-src", type=float, default=1.0)
    parser.add_argument("--w-phys", type=float, default=0.01, help="0 gives the data-only ablation")
    parser.add_argument("--hot-weight", type=float, default=0.0,
                        help="extra weight on hot pixels in the temperature loss; 0 = plain MSE")
    parser.add_argument("--phys-warmup", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--train-chips", type=int, default=0, help="0 uses all 500 training chips")
    parser.add_argument("--threads", type=int, default=0, help="0 keeps the torch default")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
