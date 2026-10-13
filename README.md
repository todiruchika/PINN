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
python generator/package_for_claude/dataset_generator.py \
    --train-size 80 --val-size 10 --test-size 10 \
    --out-dir data/raw/dataset --seed 42
```

**Verify checksums after either option:**

```bash
sha256sum -c data/SHA256SUMS
```

## Run Tests

```bash
# Validate dataset
python -m src.data.validate_dataset --dir data/raw/dataset

# Full test suite
python -m pytest tests/ -v
```

## Branch Workflow

- Work on your own branch; never commit `data/` or `.zip` files.
- Merge `main` into your branch regularly to stay current with the harness.
- Open a PR back to `main` to integrate validated results.

## Branch Status

| Branch | Purpose |
|--------|---------|
| `main` | Shared physics constants, data loaders, sensor layouts, evaluation harness only. No model training code. |
| `smriti` | Model work (smriti). |
| `Ruchika` | Model work (Ruchika). |
