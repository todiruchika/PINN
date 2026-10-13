# PINN Thermal Map

Physics-informed neural network evaluation framework for chip thermal-map reconstruction from sparse sensors.

## Repository layout

```
main branch — shared protocol, data handling, sensor layouts, evaluation harness
  Ruchika, smriti, siya — per-person method branches (PINN / ML / hybrid)
```

## Dataset setup

The dataset must be placed (or already exists) at:

```
data/raw/dataset/{train,validation,test}/{matrices,images,sensors}/
data/raw/dataset/metadata.csv
data/raw/dataset/splits.csv
```

If you have the zip file (`41c99137-dataset.zip`):

```bash
unzip 41c99137-dataset.zip -d data/raw/
```

`data/raw/` is gitignored — do not commit raw data.

## Quick start

```bash
pip install -r requirements.txt

# Validate dataset
python -m src.data.validate_dataset --dir data/raw/dataset

# Fit POD-QR sensor layout (train split only)
python -c "from src.sensors.podqr import fit_podqr_layout; from src.data.splits import load_splits; fit_podqr_layout(load_splits()['train'])"

# Generate sensor JSON files for all samples
python -m src.sensors.generate_sensors

# Run tests
python -m pytest tests/ -v
```

## Known blockers

1. **Boundary condition unknown** — Edge temperatures deviate from `T_amb_C` by up to 92 °C. The BC is not Dirichlet at T_amb.
2. **Governing equation uncertain** — May include temperature-dependent leakage (β = 0.012). Newton iterations = 1 for all samples (unconfirmed whether this matters).
3. **Thermal conductivity k not in metadata** — Cannot form the PDE residual until k is known.
4. **Only 100 samples** — Far below the ≥ 2,000 target cited in the paper.

See `PROTOCOL.md` for full details and PENDING/DECIDED status on each item.

## Branch workflow

- `main` — protocol, data loaders, sensor placement, evaluation metrics, gappy POD baseline. No model training code.
- `Ruchika`, `smriti`, `siya` (or confirmed name) — individual method implementations. Branch from `main` after this setup merges.
- Open a PR back to `main` to merge validated results into the shared harness.

## Optional: regenerate ground-truth FEM data

Requires [FEniCSx / dolfinx](https://fenicsproject.org/) (not in `requirements.txt`). See `dataset/py_pipeline/README.md` for the solver pipeline.
