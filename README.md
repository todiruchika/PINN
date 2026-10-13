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

Data is **not** committed. Every clone needs it once; it survives branch switches because `data/` is gitignored.

**Option A — copy from shared drive:**

```bash
cp -r /path/to/shared/data/raw/ data/raw/
```

**Option B — run the generator** (see `generator/README.md` for the exact command):

```bash
python generator/dataset_generator.py \
    --train-size 80 --val-size 10 --test-size 10 \
    --out-dir data/raw/dataset --seed 42
```

**Verify checksums after either option:**

```bash
sha256sum -c data/SHA256SUMS
```

**Generate POD-QR sensor readings** (required before loading with `strategy="podqr"`):

```bash
python -m src.sensors.generate_sensors
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
