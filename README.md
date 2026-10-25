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

## Real commercial-processor thermal maps (branch `github_dataset-deeponet`)

Dataset: Lu and Tan, *Thermal Map Dataset for Commercial Multi/Many Core CPU/GPU/TPU* (MLCAD 2024),
https://github.com/sheldonucr/commercial_thermal_map_dataset. No license is stated there, so the data is **not** committed:
`data/real_thermal/` is git-ignored. Download it (27 files, about 610 MB) with

```bash
python -m src.data.real_thermal --download --summary     # checksums in configs/real_thermal_sha256.txt
```

Files are pickles; they are read only through a restricted loader (`safe_load`) that accepts nothing but NumPy arrays
(a malicious pickle is blocked, see `tests/test_real_thermal.py`). Each file holds 100 consecutive frames of thermal maps in
degC plus performance-counter inputs (not power maps). Map sizes: i5 177x166, i7 185x154, Ryzen 7 4800U 223x280,
RTX 4060 250x240, Coral TPU 256x244.

Splits (`split_frames`): whole workloads for i5 (3/1/1) and i7 (13/3/3, seeded); for the three single-recording processors
contiguous blocks of 68/14/14 frames with 2 unused frames between blocks, so near-identical neighbouring frames never
straddle a boundary. Sensors: POD-QR on the training maps of each processor, same sensor-noise model as the simulated data.

```bash
python -m src.train.train_real_deeponet --processor i7 --save-maps 3      # 3 sensor counts x 5 seeds; results/real_deeponet/i7/
```

Results (test frames, non-sensor pixels, 5 seeds for the DeepONet). `rise` = relative L2 of the error against each frame's
own temperature rise above its minimum; it is the fairer scale because absolute temperatures sit near 60 to 90 degC, so the plain
relative L2 is small for any sensible map. Gappy POD (classical, number of modes chosen on validation frames) and the
training-mean map use the same sensor readings.

| Processor | Sensors | DeepONet rise | Gappy POD rise | Mean map rise | DeepONet MAE (C) |
|---|---|---|---|---|---|
| i5-3337U (300/100/100 frames) | 5 / 8 / 10 | 0.227 / 0.152 / 0.149 | **0.169 / 0.113 / 0.129** | 0.967 | 1.22 / 0.80 / 0.79 |
| i7-8650U (1300/300/300) | 5 / 8 / 10 | **0.246 / 0.105** / 0.111 | 0.367 / 0.214 / **0.086** | 0.892 | 2.42 / 1.02 / 1.09 |
| Ryzen 7 4800U (68/14/14) | 5 / 8 / 10 | 0.359 / 0.342 / 0.330 | **0.281 / 0.225 / 0.228** | 0.771 | 0.76 / 0.72 / 0.69 |
| RTX 4060 (68/14/14) | 5 / 8 / 10 | 0.176 / 0.161 / 0.177 | **0.037 / 0.027 / 0.028** | 0.740 | 2.27 / 1.98 / 2.30 |
| Coral TPU (68/14/14) | 5 / 8 / 10 | **0.362 / 0.337** / 0.360 | 0.386 / 0.367 / **0.243** | 8.06 | 0.36 / 0.33 / 0.36 |

The DeepONet beats the training-mean map everywhere and reconstructs i7 and i5 maps to about 1 to 2.4 degC mean error, but the
classical gappy POD is as good or better in most cases (clear DeepONet wins only for the i7 at 5 and 8 sensors, and the TPU at
5 and 8). The maps are low-dimensional, and the single-recording processors have only 68 training frames, where the DeepONet
is weakest (RTX 4060 hotspot error of about 175 cells). The mean map is very poor on the TPU because the test block sits later in
a drifting recording. Gappy POD is deterministic, so it has one value per setting. The DeepONet input features
(performance counters) are not used. No physics model is applied here: the heat equation, sources and boundaries are not known.

Map comparison (ground truth vs DeepONet vs gappy POD, one typical test frame per processor, the frame with the median DeepONet error):
`results/real_deeponet/comparison_maps_n8.png` and `comparison_maps_n5.png`, made by
`python -m src.eval.real_compare_maps --sensors 8 --seed 0` (retrains the DeepONet, about 10 minutes). The figures show that the RTX 4060 and
Ryzen maps contain fixed fine structure (die layout, printed text) that the DeepONet smooths away with only 68 training frames, which gappy POD
recovers because the training-mean map already contains it.

