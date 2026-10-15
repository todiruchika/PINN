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

`pred = w * DeepONet + (1 - w) * PINN` per chip, with `w` chosen per sensor count on validation chips only and
then scored once on test chips. The PINN is Ruchika's model (code copied to `pure_pinn/` from
`ruchika-per-chip-pinn`), run with a cheaper budget (600 Adam + 100 L-BFGS steps; about 3 min per solve),
1 seed. DeepONet = Model A, seed 0. Scale: **20 validation chips, 31 test chips** (the first 31 in id order),
not the full 100, because each PINN solve is slow. Raw output: `results/blend/`, `results/pinn2_metrics/`.

Test relative L2 on the same 31 chips (lower is better):

| N | DeepONet (A) | PINN (B) | Blend (w = DeepONet share) |
|---|---|---|---|
| 5  | 0.1212 | 0.1254 | **0.1194** (w=0.9) |
| 8  | 0.0946 | 0.0954 | **0.0937** (w=0.1) |
| 10 | 0.0655 | 0.0687 | **0.0641** (w=0.5) |

The blend has the lowest mean error at every sensor count, but the gains are small (1-2%) and only some are
clearly above noise (`results/blend/paired_analysis.md`): vs DeepONet at N=5, and vs PINN at N=8 and N=10. The
blend does not improve hotspot localisation (N=8, 10: worse than the DeepONet) and the physics-validity metric
stays at 0.78-0.88 for all variants. Single seed for the PINN and 31 test chips: treat this as indicative.

