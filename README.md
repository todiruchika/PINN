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

- Work on your own branch. Keep extended/scratch datasets and archive files out of Git; the canonical dataset is maintained on `main`.
- Merge `main` into your branch regularly to stay current with the harness.
- Open a PR back to `main` to integrate validated results.

## Hybrid PINN (siya branch)

This branch adds a sensor-conditioned coordinate MLP trained with supervised,
PDE-residual, and lateral Neumann-boundary losses. The model predicts the
temperature field from noisy POD-QR sensor readings. The PDE uses the shared
generator constants and the sample's `Q` map.

The POD-QR records are included in the shared dataset. Train with:

```bash
python -m src.train.hybrid_pinn --epochs 20 --n-sensors 10
```

The training command uses train and validation splits only. It writes the best
validation checkpoint to `results/hybrid_pinn/best.pt` by default. Loss weights,
model size, point counts, and other run settings are configurable through CLI
arguments; `python -m src.train.hybrid_pinn --help` lists them.

The shared data, sensor, evaluation, and physics contracts remain inherited
from `main`. Other approach branches should keep their model and training code
isolated from this branch.

Remote branches that model work goes on:

| Branch | Purpose |
|--------|---------|
| `smriti` | Model work (smriti). |
| `Ruchika` | Model work (Ruchika). |
