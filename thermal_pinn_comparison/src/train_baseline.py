"""Train Model A: a data-driven DeepONet, no physics, trained once across chips.

Unlike Model B (one physics-only solve per chip), Model A is trained on the
whole train split and evaluated once on every held-out chip -- the
"once, generalise" arm of the study, per Sec. 3.1 / 7 of the design doc.

Usage
-----
    # smoke test on the small bundled sample (20/5/5 chips)
    python src/train_baseline.py --data-root ../dataset/py_pipeline/my_dataset \
        --sensors 5 --seeds 0 --epochs 20 --quiet

    # one arm of the experiment matrix
    python src/train_baseline.py --config experiments/configs/model_a_default.json \
        --sensors 5,8,10 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow flat imports

from data_loader import GlobalScaler, chip_ids, default_data_root, grid_points, load_chip  # noqa: E402
from evaluate import append_metrics_row, field_metrics, relative_l2, sensor_rmse  # noqa: E402
from models.deeponet_baseline import build_deeponet  # noqa: E402
from utils import load_config, resolve_path, set_seed, timer  # noqa: E402

DTYPE = torch.float32
GRID = 128


# --------------------------------------------------------------------------- #
# data assembly: whole split as dense tensors, built once per (root, n_sensors)
# --------------------------------------------------------------------------- #
def build_branch_and_fields(ids: list[str], split: str, root, n_sensors: int, scaler: GlobalScaler):
    """``(branch (n, 3N) normalised, field (n, GRID*GRID) normalised)``, both float32."""
    branch = np.zeros((len(ids), 3 * n_sensors), dtype=np.float32)
    field = np.zeros((len(ids), GRID * GRID), dtype=np.float32)
    for i, sid in enumerate(ids):
        chip = load_chip(sid, split, root)
        pos, temps = chip.sensors(n_sensors)
        t_norm = scaler.normalise(temps)
        branch[i] = np.concatenate([pos, t_norm.reshape(-1, 1)], axis=1).reshape(-1).astype(np.float32)
        field[i] = scaler.normalise(chip.field()).reshape(-1).astype(np.float32)
    return branch, field


# --------------------------------------------------------------------------- #
# one training run: fixed (n_sensors, seed) -> per-test-chip metric rows
# --------------------------------------------------------------------------- #
def train_one_config(n_sensors: int, seed: int, config: dict, root, quiet: bool = True) -> dict:
    set_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    tr = config["training"]

    scaler = GlobalScaler.fit(root, split="train")
    train_ids = chip_ids(root, "train")
    val_ids = chip_ids(root, "val")
    test_ids = chip_ids(root, "test")

    train_branch, train_field = build_branch_and_fields(train_ids, "train", root, n_sensors, scaler)
    val_branch, val_field = build_branch_and_fields(val_ids, "val", root, n_sensors, scaler)

    train_branch_t = torch.tensor(train_branch, dtype=DTYPE, device=device)
    train_field_t = torch.tensor(train_field, dtype=DTYPE, device=device)
    val_branch_t = torch.tensor(val_branch, dtype=DTYPE, device=device)
    val_true_raw = scaler.denormalise(val_field).reshape(-1, GRID, GRID)

    grid_xy = torch.tensor(grid_points(GRID), dtype=DTYPE, device=device)

    model = build_deeponet({"n_sensors": n_sensors, "net": config["model"].get("net", {})}).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(tr.get("lr", 1e-3)))
    epochs = int(tr.get("epochs", 300))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(tr.get("lr_min", 1e-5)))

    batch_size = int(tr.get("batch_size", 32))
    n_query = int(tr.get("n_query", 512))
    eval_every = int(tr.get("eval_every", 10))
    patience = int(tr.get("patience", 8))

    gen = torch.Generator().manual_seed(seed)
    n_train = train_branch_t.shape[0]
    n_grid_pts = grid_xy.shape[0]

    best = {"val_rel_l2": float("inf"), "state": None, "epoch": -1}
    bad_evals = 0
    history: list[dict] = []

    with timer() as train_time:
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train, generator=gen)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n_train, batch_size):
                idx = perm[start:start + batch_size]
                b_in = train_branch_t[idx]
                q_idx = torch.randint(0, n_grid_pts, (n_query,), generator=gen)
                xy_q = grid_xy[q_idx]
                targets = train_field_t[idx][:, q_idx]

                pred = model(b_in, xy_q)
                loss = torch.mean((pred - targets) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            sched.step()

            is_last = epoch == epochs - 1
            if (epoch + 1) % eval_every == 0 or is_last:
                model.eval()
                with torch.no_grad():
                    val_pred_norm = model(val_branch_t, grid_xy).cpu().numpy()
                val_pred_raw = scaler.denormalise(val_pred_norm).reshape(-1, GRID, GRID)
                val_rel_l2 = float(np.mean([
                    relative_l2(val_pred_raw[i], val_true_raw[i]) for i in range(len(val_ids))
                ]))
                history.append({"epoch": epoch + 1, "train_loss": epoch_loss / n_batches, "val_rel_l2": val_rel_l2})
                if not quiet:
                    print(f"[A N={n_sensors} seed={seed}] epoch {epoch + 1}/{epochs} "
                          f"train_loss={epoch_loss / n_batches:.5f} val_rel_l2={val_rel_l2:.5f}")
                if val_rel_l2 < best["val_rel_l2"]:
                    best = {"val_rel_l2": val_rel_l2,
                            "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                            "epoch": epoch + 1}
                    bad_evals = 0
                else:
                    bad_evals += 1
                    if bad_evals >= patience:
                        break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    train_seconds = train_time["seconds"]

    # ---- test evaluation, one row per held-out chip ----------------------- #
    model.eval()
    rows = []
    with torch.no_grad():
        for sid in test_ids:
            chip = load_chip(sid, "test", root)
            pos, temps = chip.sensors(n_sensors)
            t_norm = scaler.normalise(temps)
            b_in = torch.tensor(
                np.concatenate([pos, t_norm.reshape(-1, 1)], axis=1).reshape(-1).astype(np.float32),
                dtype=DTYPE, device=device,
            )
            with timer() as inf_time:
                pred_norm = model.predict_field(b_in, grid_xy).cpu().numpy()
            pred = scaler.denormalise(pred_norm).reshape(GRID, GRID)
            true = chip.field()

            fm = field_metrics(pred, true, chip.T_boundary)
            row = {
                "model": "A", "bc_mode": "none", "source_kind": "", "split": "test",
                "sample_id": sid, "n_sensors": n_sensors, "seed": seed,
                **fm,
                "sensor_rmse": sensor_rmse(pred, pos, temps),
                "train_seconds": train_seconds,
                "inference_seconds": inf_time["seconds"],
            }
            rows.append(row)

    return {"rows": rows, "best_val_rel_l2": best["val_rel_l2"], "best_epoch": best["epoch"],
            "history": history, "model": model, "scaler": scaler}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model A (DeepONet baseline, no physics)")
    ap.add_argument("--config", default="experiments/configs/model_a_default.json")
    ap.add_argument("--data-root", default=None, help="defaults to ../dataset/py_pipeline/dataset")
    ap.add_argument("--sensors", default="5,8,10")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=None, help="override config training.epochs")
    ap.add_argument("--metrics-out", default="results/metrics.csv")
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    config = load_config(resolve_path(args.config))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    root = default_data_root() if args.data_root is None else resolve_path(args.data_root)

    sensors = [int(s) for s in args.sensors.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    for n_sensors in sensors:
        for seed in seeds:
            print(f"=== Model A: N={n_sensors} sensors, seed={seed} ===")
            result = train_one_config(n_sensors, seed, config, root, quiet=args.quiet)
            for row in result["rows"]:
                append_metrics_row(row, path=args.metrics_out)
            mean_rel_l2 = float(np.mean([r["rel_l2"] for r in result["rows"]]))
            print(f"    best val rel_l2={result['best_val_rel_l2']:.4f} (epoch {result['best_epoch']}) "
                  f"-> test rel_l2 mean={mean_rel_l2:.4f} over {len(result['rows'])} chips")
            if args.save_checkpoints:
                ckpt_dir = resolve_path("results/checkpoints")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(result["model"].state_dict(), ckpt_dir / f"model_a_N{n_sensors}_seed{seed}.pt")


if __name__ == "__main__":
    main()
