# Data-driven PINN: one network trained across the 500 chips

A single network is trained once on the 500 training chips and then reconstructs any
chip from its **N noisy sensor readings and its known ambient temperature**. It is the
counterpart of the per-chip PINN (`pure_pinn/` on branch `Ruchika` / `ruchika-per-chip-pinn`),
which re-fits a small network from scratch for every chip and learns nothing across chips.

|  | per-chip PINN | this model |
|---|---|---|
| learned from | one chip's sensors + the PDE | 500 chips' fields + sources + the PDE |
| per new chip | ~3 min optimisation | one forward pass (milliseconds) |
| unknown source | inferred per chip | predicted by a second head |
| training command | none (every chip is its own run) | `python -m src.train.data_driven` |

Everything is new files; nothing shared (`src/physics/heat.py`, `src/eval/`, `src/data/dataset.py`)
was modified.

---

## 1. How it works

```
sensor readings (N) + T_amb ──► branch net ──► coefficients (per chip)  ┐
                                                                         ├─► inner product ─► θ(x,y)   temperature rise
(cos πx, cos πy) ───────────────► trunk net ──► basis functions (shared) ┘                  ─► q̃(x,y)   total heat source
```

* `θ = (T − T_amb)/S` with a fixed scale `S = 50 K` (`--temp-scale`); `T = T_amb + S·θ`.
* The trunk sees `(cos πx, cos πy)`, which makes `∂θ/∂n = 0` on all four edges **exactly, for any
  weights**. The dataset's edges are adiabatic, so there is no boundary weight to tune.
* The second head predicts the **total** source `q̃ = q/(h_eff S)` (the dataset's `Q`, which already
  contains the temperature-dependent leakage), so no leakage assumption is needed.
* Loss per step, for a batch of chips:

  ```
  w_temp · MSE(θ̂, θ_true)  +  w_src · MSE(q̃̂, q̃_true)  +  w_phys · mean(R²)
  R = −ε ∇·( k(T) ∇θ̂ ) + θ̂ − q̃̂ ,     ε = t_die / (h_eff L²) ≈ 5.6e-4
  ```

  The first two are evaluated at random grid nodes of the ground-truth fields; the residual `R` is
  evaluated by autodiff at random **continuous** points, using the *predicted* source, so it
  constrains the field between grid nodes. `k(T)` is the generator's silicon conductivity.
  `w_phys` ramps up from 0 over the first 20% of training.
* **Sensor noise is re-simulated every step** from the chips' noise-free readings with the exact
  sensor model stored in the sensor files (per-sensor calibration offsets, σ = 1 °C, 1 °C
  quantisation), so the network cannot memorise one noise draw. `--no-noise-aug` trains on the
  stored noisy readings instead. **Validation and test always use the stored noisy readings**
  (`temperatures_noisy`), the protocol's model input.
* At test time only the sensor readings and `T_amb` are used. Ground truth is used only to score.

One model is trained **per sensor count** (the layout is the fixed POD-QR layout; nested 5 ⊂ 8 ⊂ 10).

### Assumptions to be aware of

* `t_die = 500 µm` and `h_eff = 6256.3 W/(m²·K)` are the generator's values, flagged
  "ASSUMPTION — NOT SOURCED" in `PROTOCOL.md`. They only enter through the physics term.
* `w_phys = 0.01` is a starting value, not a tuned one (see §5).

---

## 2. Setup

From the repository root:

```bash
pip install -r requirements.txt      # numpy pandas scipy pyyaml matplotlib pillow pytest torch
```

Data needed (all committed on `main`): `data/raw/dataset/` (fields, metadata, `splits.csv`) and
`data/generated/` (POD-QR sensor files for all 700 chips). Check first:

```bash
python -m pytest tests/test_data_driven.py -q       # 13 tests, ~20 s
```

A GPU is used automatically if PyTorch sees one (`--device auto`); `--device cpu` forces CPU.
(GPU use is supported by the code but I could only run it on CPU.)

---

## 3. Train (manual commands)

```bash
# one model: 5 sensors, seed 0, 30,000 steps
python -m src.train.data_driven --sensors 5

# the other sensor counts / seeds
python -m src.train.data_driven --sensors 8  --seed 1
python -m src.train.data_driven --sensors 10 --seed 2

# the data-only ablation: same model, no physics term (outputs go to ..._nophys)
python -m src.train.data_driven --sensors 5 --w-phys 0

# longer or shorter
python -m src.train.data_driven --sensors 5 --steps 60000
python -m src.train.data_driven --sensors 5 --steps 3000 --eval-every 500     # quick look

# continue a run that stopped (keeps the best-so-far and the loss history)
python -m src.train.data_driven --sensors 5 --resume results/data_driven_pinn/N5_seed0/last.pt

# re-score a trained model (e.g. on a different machine), optionally with pictures
python -m src.train.data_driven --sensors 5 --eval-only --out results/data_driven_pinn/N5_seed0 --save-figs 6

# the whole protocol grid, N = 5,8,10 x seeds 0..4, one after another
experiments/run_data_driven.sh
```

Progress prints every 100 steps; every 1,000 steps the model is scored on the 100 validation chips
and the best-by-validation weights are kept in `best.pt`. The **final scores use `best.pt`**,
on the validation and test splits.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--sensors N` | required | 5, 8 or 10 |
| `--seed` | 0 | model init and batch sampling |
| `--out DIR` | `results/data_driven_pinn/N<N>_seed<seed>[_nophys]` | where everything is written |
| `--steps` | 30000 | optimiser steps |
| `--batch-chips` | 32 | chips per step |
| `--points-data` / `--points-phys` | 1024 / 256 | supervised grid points / PDE points per chip per step |
| `--lr`, `--lr-min` | 1e-3, 1e-5 | Adam, cosine decay |
| `--grad-clip` | 1.0 | 0 disables |
| `--w-temp`, `--w-src`, `--w-phys` | 1, 1, 0.01 | loss weights; `--w-phys 0` = data-driven ablation |
| `--phys-warmup` | 0.2 | fraction of steps to ramp `w-phys` up |
| `--no-noise-aug` | off | train on the stored noisy readings only |
| `--latent` | 64 | branch/trunk inner-product size |
| `--branch-width/-depth`, `--trunk-width/-depth` | 128/3, 128/4 | network size (≈118 k parameters) |
| `--activation` | tanh | tanh, silu, gelu |
| `--temp-scale` | 50 | θ scale in K |
| `--soft-neumann` | off | raw `(x, y)` into the trunk: edges no longer exactly adiabatic |
| `--eval-every` | 1000 | validation interval |
| `--eval-chips` | 0 (all 100) | validation chips used *during training* |
| `--eval-splits` | `validation,test` | splits scored at the end |
| `--save-figs K` | 0 | true / predicted / error pictures for K chips per split |
| `--resume CKPT` | | continue from a checkpoint |
| `--eval-only`, `--ckpt` | | skip training and score a checkpoint (default `<out>/best.pt`) |

### Outputs (in `--out`)

| File | Content |
|---|---|
| `best.pt`, `last.pt` | checkpoints (`*.pt` is git-ignored); each carries its own model config |
| `history.csv`, `curves.png` | training loss terms and validation error against step |
| `per_chip_validation.csv`, `per_chip_test.csv` | one row per chip |
| `seed_<seed>.csv` | one row: chip-averaged test scores, in the layout `src/eval/report.py` aggregates |
| `summary.json`, `args.json` | scores and the exact settings |
| `figures/<split>/…png` | with `--save-figs` |

Scores use `src/eval/metrics.py` on the **non-sensor nodes only** (protocol), in °C:
`relative_l2`, `mae_C`, `rmse_C`, `mse_C2`, `peak_temp_error_C` (signed) and `peak_temp_abs_error_C`,
`hotspot_localization_cells`, plus `source_relative_l2` for the predicted source field.

Summarise several seeds with `python experiments/aggregate_data_driven.py` (mean ± std over seeds,
grouped by `config`, e.g. `data_driven_pinn_N5` vs `data_driven_pinn_N5_nophys`).

---

## 4. Cost

Measured on this CPU-only 4-core machine while other jobs were running: about **0.2 s per step**
with the physics term and a little less without it, so the default 30,000 steps is on the order of
**1–2 hours per model on a busy CPU**, and the full grid (3 sensor counts × 5 seeds) about a day.
A GPU should be much faster. Inference is a single forward pass: milliseconds per chip.

---

## 5. Notes on use

* **Comparable with the per-chip PINN** only on the same chips and metrics. Both report relative L2 on
  non-sensor nodes in °C; the per-chip PINN was only run on a handful of chips so far.
* **Do not tune on the test split.** Choose `w-phys`, `--steps`, etc. from the validation scores
  (`per_chip_validation.csv`, `history.csv`), then read the test scores once.
* Run the `--w-phys 0` ablation to see what the physics term adds; that comparison is the main
  scientific question for this model.
