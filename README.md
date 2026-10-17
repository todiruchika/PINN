# PINN Thermal Map

## Setup

```bash
git clone <repo-url>
cd PINN
git checkout -b <your-name>-<topic>   # branch from main
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The canonical 700-sample dataset (500 train / 100 validation / 100 test) and its
POD-QR sensor records are committed under
`data/raw/dataset/` and `data/generated/`. Clone or pull the repository to get the
same train/validation/test data on every approach branch. The incomplete extended
and scratch datasets remain local and are not committed.

**Verify the dataset checksums:**

```bash
sha256sum -c data/SHA256SUMS
```

## Run Tests

```bash
# Validate dataset
python -m src.data.validate_dataset --dir data/raw/dataset

# Full test suite
python -m pytest tests/ -v

# Coverage of delivered 100 vs data/raw_extended/ (writes results/validation/coverage_report.md)
python -m src.eval.coverage
```

## Branch Workflow

- Work on your own branch; never commit `data/` or `.zip` files.
- Merge `main` into your branch regularly to stay current with the harness.
- Open a PR back to `main` to integrate validated results.

## Branch Status

`main` contains:

- `src/data/`: dataset loader, splits, dataset validator
- `src/sensors/`: sensor layouts, POD-QR placement, readers
- `src/eval/`: metrics, gappy POD baseline, report aggregator, coverage check
- `src/physics/heat.py`: stubs that raise `NotImplementedError`
- `tests/`: pytest suite
- No model or training code (`src/train/` is an empty package)

Remote branches that model work goes on:

| Branch | Purpose |
|--------|---------|
| `smriti` | Model work (smriti). |
| `Ruchika` | Model work (Ruchika). |

## Model A: DeepONet baseline (branch `smriti`)

Pure data-driven DeepONet, no physics term, on the current 500/100/100 dataset
(`data/raw/dataset`). Branch net: sensor positions + `temperatures_noisy`;
trunk net: query coordinate. Fields are scaled as `(T - T_amb) / dT_scale`
(global train scale). Metrics are on non-sensor nodes (`src/eval/metrics.py`).

```bash
python -m src.sensors.generate_sensors            # only needed if data/generated is missing (it is committed on main)
python -m src.train.train_deeponet                # configs/deeponet.yaml: N in {5,8,10} x seeds 0-4
python -m src.train.train_deeponet --sensors 5 --seeds 0 --epochs 20   # smoke test
```

Code: `src/models/deeponet.py`, `src/train/train_deeponet.py`, `configs/deeponet.yaml`,
`tests/test_deeponet.py`. Results: `results/model_a/` (`summary.md`, `seed_*.csv`, `per_sample_*.csv`).

Test results, mean +/- std over 5 seeds (CPU, ~2 min per run), POD-QR layout as refit on the
500 train samples:

| N sensors | rel. L2 | MAE (C) | RMSE (C) |
|---|---|---|---|
| 5  | 0.1146 +/- 0.0010 | 4.66 +/- 0.06 | 6.55 +/- 0.05 |
| 8  | 0.0862 +/- 0.0006 | 3.37 +/- 0.03 | 4.85 +/- 0.04 |
| 10 | 0.0613 +/- 0.0003 | 2.38 +/- 0.02 | 3.45 +/- 0.02 |

Thermal maps (true | prediction | abs. error, first 3 test samples) are saved as
`results/model_a/maps_n{N}_seed{k}.png` (`--save-maps 3`).

## Model C: hybrid physics-informed DeepONet (branch `smriti-hybrid`)

Same data, split, sensors, scaling, seeds and metrics as Model A. Full table: `results/model_c_summary.md`.

* **Stage 1** (`src/models/hybrid.py`, `src/train/train_hybrid.py`): DeepONet with a second, non-negative
  source output and exactly adiabatic edges (trunk input `cos(pi x), cos(pi y)`); optional heat-equation
  residual term from `src/physics/losses.py` (copied from `ruchika-per-chip-pinn`).
* **Stage 2** (`src/train/refine_hybrid.py`): per-chip refinement at test time. Fit Gaussian power blobs to the
  source the stage-1 map implies, then tune the chip's 128 coefficients on sensor fit + heat-equation residual
  + closeness to stage 1. Settings were chosen on validation chips only.
* **Physics-validity metric** (`src/physics/residual.py`): relative finite-difference residual of the governing
  equation against the dataset's true source `Q`. Ground truth scores about 0.15 (discretisation floor).

```bash
python -m src.train.train_hybrid --config configs/hybrid_pretrain.yaml --save-checkpoints --out-dir results/model_c_pretrain/n5 --sensors 5
python -m src.train.refine_hybrid --split test --sensors 5 --seeds 0 --w-prior 1 --w-pde 10 --w-data 10
```

**Result: the hybrid does not beat Model A here.** Test rel. L2 (mean over 5 seeds for A and stage 1):

| N | Model A | C stage 1 | C stage 2 (seed 0) |
|---|---|---|---|
| 5  | 0.1146 | 0.1147 | 0.1188 (stage 1 same seed: 0.1130) |
| 8  | 0.0862 | 0.0861 | 0.0894 (0.0860) |
| 10 | 0.0613 | 0.0616 | 0.0678 (0.0623) |

Validation sweeps (N=5, seed 0) found the same thing: adding the residual term during training gave
0.1183 (weight 0), 0.1177 (0.1), 0.1240 (1), 0.1419 (10); supplying the true power map as extra labels gave
about 0.116. Refinement was worse in all 9 settings tried and worse with a larger physics weight.
Predicted maps are far from satisfying the equation for every model (residual 0.79 to 0.86 vs 0.15 floor) and
the physics terms did not reduce it. Likely cause: the source is unknown, so the heat equation does not pin the
temperature down, and 500 training maps already give Model A plenty of data. Untested ideas: fewer training
chips, a better source prior.

### Model C, variant 2: blend of DeepONet and per-chip PINN (`src/eval/blend.py`)

`pred = w * DeepONet + (1 - w) * PINN` per chip. `w` is chosen per sensor count on the **100 validation chips**
(grid 0, 0.1, ..., 1, minimum mean relative L2), then locked and scored once on the **100 test chips**. The PINN is
Ruchika's model (code copied to `pure_pinn/` from `ruchika-per-chip-pinn`), run with a cheaper budget than her default
(600 Adam + 100 L-BFGS steps, about 3-5 min per solve), 1 seed, on every validation and test chip (600 solves).
DeepONet = Model A, seed 0. Raw output: `results/blend/`, `results/pinn2_metrics/`.

Test relative L2, all 100 test chips, same chips for every model (lower is better):

| N | DeepONet (A) | PINN (B) | Blend | chosen w (DeepONet share) |
|---|---|---|---|---|
| 5  | 0.1159 | 0.1333 | 0.1162 | 0.8 |
| 8  | 0.0859 | 0.0880 | **0.0821** | 0.5 |
| 10 | 0.0616 | 0.0651 | **0.0601** | 0.5 |

Paired comparison over the 100 chips (`results/blend/paired_analysis.md`, 95% bootstrap CI of the mean difference):

* Blend vs PINN: clearly better at all three sensor counts.
* Blend vs DeepONet: clearly better at N=8 (-4.4% error, CI above zero); at N=10 it is 2.4% lower but the CI includes
  zero; at N=5 it is a tie (0.1162 vs 0.1159).

So the blend is never worse than either model by a meaningful amount and is clearly better than the DeepONet in only
one of three settings. The PINN alone is worse than the DeepONet at every N on the full test set. Hotspot localisation
is not improved by the blend, and the physics-validity metric is 0.76-0.87 for every variant. Caveats: 1 PINN seed, the
shortened PINN budget, and 1 DeepONet seed in the blend. An early look at only the first 31 test chips showed a larger,
1-2% blend advantage at every N; it did not hold on the full set.

### Model C, low-data experiment (`results/low_data/`)

Same models and test chips, but the DeepONet is trained on only 25, 50 or 100 chips (nested seeded subsets; epochs scaled so
the number of gradient steps matches the 500-chip run). The PINN needs no training chips, so its results are unchanged. The blend
weight is chosen on the 100 validation chips as before. Full table with confidence intervals: `results/low_data/summary.md`.
Run: `python -m src.train.train_deeponet --n-train 25 --save-checkpoints --out-dir results/low_data/n25`, then
`python -m src.eval.blend --n-train 25 --ckpt-dir results/low_data/n25/checkpoints --out results/low_data/blend_n25`.

Test relative L2 on the 100 test chips (DeepONet = mean of 5 seeds):

| N | training chips | DeepONet | PINN | Blend |
|---|---|---|---|---|
| 5  | 25  | 0.1383 | 0.1333 | **0.1276** |
| 5  | 50  | 0.1334 | 0.1333 | **0.1251** |
| 5  | 100 | 0.1244 | 0.1333 | **0.1221** |
| 5  | 500 | **0.1146** | 0.1333 | 0.1162 |
| 8  | 25  | 0.1274 | 0.0880 | **0.0878** |
| 8  | 50  | 0.1134 | 0.0880 | **0.0872** |
| 8  | 100 | 0.0952 | 0.0880 | **0.0853** |
| 8  | 500 | 0.0862 | 0.0880 | **0.0821** |
| 10 | 25  | 0.0933 | 0.0651 | **0.0644** |
| 10 | 50  | 0.0726 | 0.0651 | **0.0623** |
| 10 | 100 | 0.0642 | 0.0651 | **0.0611** |
| 10 | 500 | 0.0613 | 0.0651 | **0.0601** |

With little training data the DeepONet degrades sharply while the PINN does not, and the blend's advantage over the DeepONet grows
(for example N=8, 25 chips: 0.1274 -> 0.0878, about 31% lower; all N=8 differences have a 95% interval above zero). At 500 chips
the blend and the DeepONet tie at N=5 and N=10. Caveats: at the smallest training sets the blend is close to the PINN alone
(N=8, 25 chips: 0.0878 vs 0.0880), so most of the gain comes from the PINN rather than from combining; the POD-QR sensor
layout was fit on all 500 training chips and kept fixed; and the 100 validation chips (used for early stopping and blend
weights) would not be available in a truly data-scarce setting.

