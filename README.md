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
