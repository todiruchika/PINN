"""Run Ruchika's per-chip PINN (pure_pinn/src, branch pure-pinn) on *our* real-thermal test/validation frames.

The PINN learns nothing across frames, so any frame can be used. To make it comparable with the DeepONet it is given
exactly the same noisy sensor readings (same sensor pixels, same noise) as the DeepONet gets. Frames are a regular subsample
(``subset_idx``) because every solve takes minutes. Her ``train_one_chip`` is used unchanged; a small stand-in object plays the role of her
``ChipSample``. The die has no known physical size, so the unit-square config ``model_b_github.json`` is used, and the
ambient temperature (the sink temperature in the equation) is set to the minimum of the training maps of that processor.

    python -m src.eval.real_pinn_runs --processor i5 --shard 0 --shards 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path("pure_pinn/src").resolve()))      # her modules use flat imports
from train_pinn import train_one_chip                          # noqa: E402

from src.train.train_real_deeponet import prepare, read_sensors  # noqa: E402

OUT = Path("results/real_pinn")
SUBSET = {"i5": 10, "i7": 20}          # frames per split


def subset_idx(n_frames: int, k: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_frames - 1, k).round().astype(int))


class RealChip:
    """Duck-typed stand-in for ChipSample: noisy readings in, ground-truth field kept for scoring only."""

    def __init__(self, sample_id, split, true_map, positions, readings, t_amb):
        self.sample_id, self.split, self.T_amb, self.die_L = sample_id, split, float(t_amb), 1.0
        self.peak_temperature, self.min_temperature = float(true_map.max()), float(true_map.min())
        self.block_count, self.blocks, self.root = 0, [], Path("__no_such_root__")
        self._map, self._pos, self._read = true_map.astype(np.float64), positions, readings

    def sensors(self, n):
        return self._pos[:n], self._read[:n]

    def field(self):
        return self._map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", required=True, choices=["i5", "i7"])
    ap.add_argument("--config", default="configs/real_deeponet.yaml")
    ap.add_argument("--pinn-config", default="pure_pinn/experiments/configs/model_b_github.json")
    ap.add_argument("--adam", type=int, default=1000)
    ap.add_argument("--lbfgs", type=int, default=200)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    pc = json.load(open(a.pinn_config))
    pc["training"].update(adam_steps=a.adam, lbfgs_steps=a.lbfgs)
    D = prepare(a.processor, cfg)
    H, W = D["shape"]
    t_amb = float(D["maps"]["train"].min())
    out = OUT / a.processor; out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split in ("validation", "test"):
        idx = subset_idx(len(D["maps"][split]), SUBSET[a.processor])
        for n in cfg["sensor_counts"]:
            for i in idx:
                jobs.append((split, int(i), n))
    mine = [j for k, j in enumerate(jobs) if k % a.shards == a.shard]
    print(f"{a.processor}: {len(jobs)} solves in total, {len(mine)} in this shard, T_amb={t_amb:.2f}", flush=True)
    rows, cache = [], {}
    for split, i, n in mine:
        f = out / f"{split}_{i:03d}_N{n}.npy"
        if f.exists():
            continue
        key = (split, n)
        if key not in cache:                      # same readings as the DeepONet (deterministic noise)
            first = D["frames"][split][0][1] if len({w for w, _ in D["frames"][split]}) == 1 else 0
            cache[key] = read_sensors(D["maps"][split], D["rc"], first, n, cfg, ("train", "validation", "test").index(split))
        pos = np.stack([D["rc"][:n, 1] / (W - 1), D["rc"][:n, 0] / (H - 1)], axis=1)
        chip = RealChip(f"{a.processor}_{split}_{i:03d}", split, D["maps"][split][i], pos, cache[key][i], t_amb)
        res = train_one_chip(chip, n, 0, pc, verbose=False)
        np.save(f, res["pred_field"].astype(np.float32))
        rows.append({**res["row"], "frame": i})
        pd.DataFrame(rows).to_csv(out / f"metrics_shard{a.shard}.csv", index=False)
        print(f"  {split} frame {i} N={n}: rel_l2={res['row']['rel_l2']:.4f} ({res['row']['train_seconds']:.0f}s)", flush=True)
    print("shard done", flush=True)


if __name__ == "__main__":
    main()
