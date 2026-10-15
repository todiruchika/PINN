"""Blend Model A (DeepONet) and Model B (per-chip PINN) predictions.

    pred = w * DeepONet + (1 - w) * PINN        (in degC, per chip)

For each sensor count, ``w`` is chosen on the *validation* chips only (grid
0, 0.1, ..., 1 minimising mean relative L2 on non-sensor nodes), then locked
and scored once on the *test* chips. Only chips for which a PINN solve exists
are used, and every model is scored on exactly that same set of chips.

PINN predictions are the ``*_pred.npy`` files written by
``pure_pinn/src/train_pinn.py --save-fields`` into ``results/pinn2_runs/``.
DeepONet predictions come from the seed-0 checkpoints in
``results/model_a_ckpt/**/checkpoints/``.

Usage
-----
    python -m src.eval.blend
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import GRID, RAW_DIR, load_metadata
from src.data.splits import load_splits
from src.eval.metrics import all_metrics
from src.models.deeponet import build_deeponet
from src.physics.residual import relative_pde_residual
from src.train.train_deeponet import branch_inputs, grid_points, load_split

PINN_RUNS = Path("results/pinn2_runs")
CKPT_DIRS = [Path("results/model_a_ckpt/checkpoints"), Path("results/model_a_ckpt/b/checkpoints")]
OUT = Path("results/blend")
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def find_pinn_pred(sample_id: str, n: int) -> np.ndarray | None:
    hits = sorted(PINN_RUNS.glob(f"{sample_id}_N{n}_seed0_*_pred.npy"))
    return np.load(hits[0]).astype(np.float64) if hits else None


def deeponet_preds(config: dict, n: int, data: dict, split: str) -> np.ndarray:
    model = build_deeponet({"n_sensors": n, "net": config["model"]})
    path = next(d / f"deeponet_n{n}_seed0.pt" for d in CKPT_DIRS if (d / f"deeponet_n{n}_seed0.pt").exists())
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    tr = data["train"]
    scale = float((tr["T"] - tr["T_amb"][:, None, None]).max())
    d = data[split]
    b = torch.tensor(branch_inputs(d, scale), dtype=torch.float32)
    with torch.no_grad():
        theta = model(b, torch.tensor(grid_points(), dtype=torch.float32)).numpy()
    return theta.reshape(-1, GRID, GRID).astype(np.float64) * scale + d["T_amb"][:, None, None]


def score(pred: np.ndarray, d: dict, i: int) -> dict:
    return {**all_metrics(pred, d["T"][i], d["mask"][i]),
            "pde_residual_rel": relative_pde_residual(pred, d["Q"][i], d["T_amb"][i])}


def main() -> None:
    config = yaml.safe_load(open("configs/deeponet.yaml"))
    meta, splits = load_metadata(RAW_DIR), load_splits(RAW_DIR / "splits.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, chosen = [], {}
    for n in (5, 8, 10):
        data = {sp: load_split(splits[sp], n, config["sensor_strategy"], meta) for sp in ("train", "validation", "test")}
        preds = {}
        for sp in ("validation", "test"):
            A = deeponet_preds(config, n, data, sp)
            keep = []
            for i, sid in enumerate(data[sp]["ids"]):
                P = find_pinn_pred(sid, n)
                if P is not None:
                    keep.append((i, A[i], P))
            preds[sp] = keep
        if not preds["validation"] or not preds["test"]:
            print(f"N={n}: not enough finished PINN solves (val={len(preds['validation'])}, test={len(preds['test'])})")
            continue
        # choose w on validation chips only
        val = data["validation"]
        curve = {}
        for w in WEIGHTS:
            curve[w] = np.mean([score(w * A + (1 - w) * P, val, i)["relative_l2"] for i, A, P in preds["validation"]])
        w_best = min(curve, key=curve.get)
        chosen[n] = (w_best, curve)
        print(f"N={n}: w (DeepONet share) = {w_best}  val rel_l2 by w: " +
              " ".join(f"{w:.1f}:{v:.4f}" for w, v in curve.items()))
        te = data["test"]
        for name, f in (("DeepONet (A)", lambda A, P: A), ("PINN (B)", lambda A, P: P),
                        (f"Blend w={w_best}", lambda A, P: w_best * A + (1 - w_best) * P)):
            for i, A, P in preds["test"]:
                rows.append({"n_sensors": n, "model": name.split(" w=")[0], "sample_id": te["ids"][i], **score(f(A, P), te, i)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_chip_test.csv", index=False)
    summ = df.groupby(["n_sensors", "model"])[["relative_l2", "mae_C", "rmse_C", "peak_temp_error_C",
                                                "hotspot_localization_cells", "pde_residual_rel"]].mean()
    summ["n_chips"] = df.groupby(["n_sensors", "model"]).size()
    order = {"DeepONet (A)": 0, "PINN (B)": 1, "Blend": 2}
    summ = summ.reset_index().sort_values(["n_sensors", "model"], key=lambda c: c.map(order) if c.name == "model" else c)
    lines = [f"Blend weight (DeepONet share) chosen on validation chips: " +
             ", ".join(f"N={n}: {w}" for n, (w, _) in chosen.items()), "", summ.round(4).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
