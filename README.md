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

The canonical 100-sample dataset and its POD-QR sensor records are committed under
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
python -m src.sensors.generate_sensors            # POD-QR sensor files for all 700 samples (local, not committed)
python -m src.train.train_deeponet                # configs/deeponet.yaml: N in {5,8,10} x seeds 0-4
python -m src.train.train_deeponet --sensors 5 --seeds 0 --epochs 20   # smoke test
```

Code: `src/models/deeponet.py`, `src/train/train_deeponet.py`, `configs/deeponet.yaml`,
`tests/test_deeponet.py`. Results: `results/model_a/` (`summary.md`, `seed_*.csv`, `per_sample_*.csv`).

Test results, mean +/- std over 5 seeds (CPU, ~1.7 min per run):

| N sensors | rel. L2 | MAE (C) | RMSE (C) |
|---|---|---|---|
| 5  | 0.1134 +/- 0.0006 | 4.60 +/- 0.04 | 6.46 +/- 0.03 |
| 8  | 0.0822 +/- 0.0007 | 3.24 +/- 0.03 | 4.69 +/- 0.04 |
| 10 | 0.0662 +/- 0.0001 | 2.58 +/- 0.01 | 3.73 +/- 0.01 |

The POD-QR layout is the one already on `main` (fit on the earlier 80-sample train
split); it was not refit on the 500-sample train split.
