"""Per-frame FBPINN on the real thermal maps: Ruchika's PINN pipeline with the field network swapped for an FBPINN.

Everything else is as in ``real_pinn_runs.py``: her ``train_one_chip`` unchanged, config ``model_b_github.json``, budget 600 Adam +
100 L-BFGS steps, one seed, ``T_amb`` = minimum of the training maps, and exactly the same sampled frames (``subset_idx``, ``SUBSET``)
and noisy sensor readings as the DeepONet and the PINN. Field network: ``src/models/real_fbpinn.py``.

    python -m src.eval.real_fbpinn_runs --processor i5 --shard 0 --shards 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eval.real_pinn_runs import SUBSET, RealChip, subset_idx       # also puts pure_pinn/src on sys.path
import train_pinn                                                        # noqa: E402  (her module, flat import)
from src.models.real_fbpinn import build_fbpinn                          # noqa: E402
from src.train.train_real_deeponet import prepare, read_sensors          # noqa: E402

OUT = Path("results/real_fbpinn")
PREDS = Path("results/real_new_models/preds/fbpinn")                     # git-ignored

train_pinn.build_pinn = build_fbpinn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", required=True, choices=["i5", "i7"])
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--pinn-config", default="pure_pinn/experiments/configs/model_b_github.json")
    ap.add_argument("--adam", type=int, default=600)
    ap.add_argument("--lbfgs", type=int, default=100)
    ap.add_argument("--sensors", default=None)
    ap.add_argument("--frames", type=int, default=None, help="smoke test: only the first k sampled frames per split")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    pc = json.load(open(a.pinn_config))
    pc["training"].update(adam_steps=a.adam, lbfgs_steps=a.lbfgs)
    sensors = [int(s) for s in a.sensors.split(",")] if a.sensors else cfg["sensor_counts"]
    smoke = a.frames is not None or (a.adam, a.lbfgs) != (600, 100)
    D = prepare(a.processor, cfg)
    H, W = D["shape"]
    t_amb = float(D["maps"]["train"].min())
    out = (Path("results/real_new_models/smoke/fbpinn") if smoke else OUT) / a.processor
    preds = (out if smoke else PREDS / a.processor)
    out.mkdir(parents=True, exist_ok=True); preds.mkdir(parents=True, exist_ok=True)
    jobs = []                                    # sensor count outermost: each count completes (validation + test) as a block
    for n in sensors:
        for split in ("validation", "test"):
            jobs += [(split, int(i), n) for i in subset_idx(len(D["maps"][split]), SUBSET[a.processor])[: a.frames]]
    mine = [j for k, j in enumerate(jobs) if k % a.shards == a.shard]
    print(f"{a.processor}: {len(jobs)} FBPINN solves in total, {len(mine)} in this shard, T_amb={t_amb:.2f}", flush=True)
    log = out / f"metrics_shard{a.shard}.csv"
    rows, cache = (pd.read_csv(log).to_dict("records") if log.exists() else []), {}
    for split, i, n in mine:
        f = preds / f"{split}_{i:03d}_N{n}.npy"
        if f.exists():
            continue
        if (split, n) not in cache:              # same readings as the DeepONet (deterministic noise)
            first = D["frames"][split][0][1] if len({w for w, _ in D["frames"][split]}) == 1 else 0
            cache[(split, n)] = read_sensors(D["maps"][split], D["rc"], first, n, cfg, ("train", "validation", "test").index(split))
        pos = np.stack([D["rc"][:n, 1] / (W - 1), D["rc"][:n, 0] / (H - 1)], axis=1)
        chip = RealChip(f"{a.processor}_{split}_{i:03d}", split, D["maps"][split][i], pos, cache[(split, n)][i], t_amb)
        res = train_pinn.train_one_chip(chip, n, 0, pc, verbose=False)
        np.save(f, res["pred_field"].astype(np.float32))
        rows.append({**res["row"], "model": "FBPINN", "frame": i})
        pd.DataFrame(rows).to_csv(log, index=False)
        print(f"  {split} frame {i} N={n}: rel_l2={res['row']['rel_l2']:.4f} ({res['row']['train_seconds']:.0f}s, "
              f"{res['row']['n_params']} params)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
