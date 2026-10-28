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
`data/real_thermal/` is git-ignored. Download the two processors used here (about 500 MB) with

```bash
python -m src.data.real_thermal --download --summary     # checksums in configs/real_thermal_sha256.txt
```

**Scope.** The study uses the two Intel CPUs only: **i5-3337U** and **i7-8650U**. The Ryzen 7 4800U and RTX 4060 were dropped (one 100-frame
recording each, only 68 training frames, and maps with fixed fine structure such as die layout and printed text that smooth models cannot
reproduce); the Coral TPU is set aside. Their code path and results are kept under `results/real_deeponet/_not_used/` and can still be
downloaded with `--processors`.

Files are pickles, read only through a restricted loader (`safe_load`) that accepts nothing but NumPy arrays (a malicious pickle is blocked, see
`tests/test_real_thermal.py`). Each file holds 100 consecutive frames of thermal maps in degC plus performance-counter inputs (not power maps,
and not used). Map sizes: i5 177x166, i7 185x154.

**Splits** (`split_frames`, seeded): whole workloads go to one split, so no test frame has a near-identical neighbour in training.

| Processor | Train | Validation | Test |
|---|---|---|---|
| i5-3337U (5 workloads x 100 frames) | 3 workloads = 300 frames | 1 workload (stream) = 100 | 1 workload (flac) = 100 |
| i7-8650U (19 workloads x 100 frames) | 13 workloads = 1300 frames | 3 (aio-stress, mbw, ramspeed) = 300 | 3 (build-gcc, cachebench, git) = 300 |

The exact test frames and the POD-QR sensor pixels are in `results/real_deeponet/splits_and_sensors_i5_i7.json`, so another model (for example a PINN)
can be scored on exactly the same frames and sensors. Sensors: POD-QR on each processor's training maps, nested 5 / 8 / 10, 3 pixels from
the edge; readings = map value + fixed per-sensor offset (std 0.5 C) + Gaussian noise (std 1 C), rounded to 1 C.

```bash
python -m src.train.train_real_deeponet --processor i7 --save-maps 3      # 3 sensor counts x 5 seeds -> results/real_deeponet/i7/
python -m src.train.train_real_deeponet --processor i5 --save-maps 3
python -m src.eval.real_compare_maps --processors i5,i7 --sensors 8 --out results/real_deeponet/comparison_maps_i5_i7_n8.png
```

**Results** (test frames, non-sensor pixels; DeepONet = mean over 5 seeds; `rise` = relative error against each frame's own temperature rise above its
minimum, the fairer scale because absolute temperatures sit near 60 to 90 degC; gappy POD = classical reconstruction from the same sensor readings, number
of modes chosen on validation frames; `no sensors` = the training-mean map). Full table: `results/real_deeponet/summary_i5_i7.csv`; chart:
`results/real_deeponet/results_i5_i7.png`.

| Processor | Sensors | Model | rise error | MAE (C) | RMSE (C) | Peak error (C) | Hotspot off by (px) |
|---|---|---|---|---|---|---|---|
| i5-3337U | 5 | DeepONet | 0.227 ± 0.022 | 1.22 ± 0.15 | 1.45 | -1.7 | 19.6 |
| | | gappy POD | **0.169** | **0.91** | 1.06 | +0.4 | 15.7 |
| | 8 | DeepONet | 0.152 ± 0.013 | 0.80 ± 0.07 | 0.98 | -1.2 | 18.9 |
| | | gappy POD | **0.113** | **0.62** | 0.72 | +0.3 | 14.7 |
| | 10 | DeepONet | 0.149 ± 0.017 | 0.79 ± 0.10 | 0.97 | -1.1 | 16.4 |
| | | gappy POD | **0.129** | **0.69** | 0.83 | +0.5 | 21.4 |
| | any | no sensors | 0.967 | 6.07 | 6.31 | -7.6 | 16.2 |
| i7-8650U | 5 | DeepONet | **0.246 ± 0.005** | **2.42 ± 0.05** | 3.09 | -6.4 | 20.9 |
| | | gappy POD | 0.367 | 3.86 | 4.92 | -9.2 | 25.2 |
| | 8 | DeepONet | **0.105 ± 0.009** | **1.02 ± 0.10** | 1.27 | -2.2 | 9.3 |
| | | gappy POD | 0.214 | 2.26 | 2.85 | -6.2 | 13.9 |
| | 10 | DeepONet | 0.111 ± 0.045 | 1.09 ± 0.46 | 1.40 | -3.2 | 11.8 |
| | | gappy POD | **0.086** | **0.85** | 1.03 | -0.1 | 10.8 |
| | any | no sensors | 0.892 | 10.43 | 11.63 | -16.7 | 55.6 |

How to read it. Both methods recover the maps to about 1 to 2.4 degC mean error, far better than using no sensors (6 to 10 degC). On the **i7**, the
DeepONet is clearly better with 5 and 8 sensors (it learns from 1300 training frames) and slightly worse with 10, where one of the 5 seeds was
noticeably poorer (rise error 0.111 ± 0.045). On the **i5** the classical gappy POD is better at every sensor count. Both models under-predict the
peak temperature (the DeepONet by 1 to 6 degC). Caveats: the i5 numbers rest on a single test workload (and a single validation workload); gappy POD is
deterministic, so it has one value per setting; the DeepONet was trained on 300 (i5) and 1300 (i7) frames while a PINN uses none, so the training
data are not comparable between methods, only the test frames are. No physics model is applied here: the heat equation, sources and boundaries
of these processors are not known.

Map comparisons (truth, DeepONet, gappy POD, errors; the frame with the median DeepONet error, seed 0): `results/real_deeponet/comparison_maps_i5_i7_n5.png`,
`_n8.png` and `_n10.png`.

### Hybrid: DeepONet + physics (branch `github_dataset-hybrid`)

No PINN for this dataset was on GitHub yet, so the physics side is a **physics-based inverse reconstruction** (`src/physics/screened_poisson.py`), a
PINN-style model without a neural network. It assumes steady heat conduction with vertical heat loss and adiabatic edges,
`-kappa*laplacian(T) + h*(T - T0) = q`, with unknown heat sources `q` written as Gaussian bumps on a coarse grid; the equation is solved
for each bump with a cosine transform and the bump amplitudes are fitted to the noisy sensor readings with a ridge penalty. It uses no training maps. Its
decay length, bump spacing and ridge are chosen on validation frames. It can be swapped for a neural PINN: the interface is just "map predictions for the same test frames".

Hybrids (`src/eval/real_hybrid.py`; summary `src/eval/real_hybrid_report.py`; 5 seeds x 3 sensor counts x i5/i7):
1. **blend**: `w*DeepONet + (1-w)*Physics`, w chosen on validation;
2. **DeepONet + physics correction**: fit the physics response to the DeepONet's residual at the sensors and add it to the DeepONet map;
3. blend, then correction.

**"The hybrid" is variant 2**, picked as the variant with the lowest mean validation error over all settings (validation only, no test frames in the choice;
variants 1 and 3 are within noise of each other on validation). An earlier rule (best variant per setting on validation) was dropped because validation could not
tell the variants apart. I had already seen per-seed test numbers when I wrote the second rule, so treat the choice as made with some knowledge of the results.

Test frames, relative error against the temperature rise (mean of 5 seeds; lower is better; `results/real_hybrid/`):

| Processor | Sensors | DeepONet | Physics | **Hybrid (DeepONet + physics correction)** | Hybrid vs DeepONet | Gappy POD (ref.) |
|---|---|---|---|---|---|---|
| i5-3337U | 5 | **0.227** | 0.419 | 0.256 | worse | 0.169 |
| | 8 | 0.152 | 0.248 | 0.149 | tie | 0.113 |
| | 10 | 0.149 | 0.158 | **0.141** | better | 0.129 |
| i7-8650U | 5 | 0.246 | 0.501 | 0.246 | tie | 0.367 |
| | 8 | 0.104 | 0.332 | **0.101** | better | 0.214 |
| | 10 | 0.111 | 0.284 | **0.099** | better | 0.086 |

"better / worse / tie" is a paired 95% interval over the test frames (`results/real_hybrid/paired_comparison.csv`). The hybrid is clearly better than the physics model in
all six settings, and better than the DeepONet in 3 of 6 (tie in 2, worse in 1: i5 with 5 sensors). So it is better than **both** individual models in 3 of 6 settings, by
small margins (3 to 11% lower error), and the classical gappy POD is still better on the i5 and at 10 sensors on the i7. The physics model alone is much weaker than the
DeepONet, so blending it in (variant 1) does not help; the useful part is the physics-based correction at the sensors. Caveats: the i5 has a single validation and a single test workload and
its validation frames are much harder than its test frames (DeepONet validation error 0.5 vs test 0.15), so validation-based choices transfer imperfectly; frames inside a workload are
strongly correlated, so the frame-level intervals are optimistic; the physics model is not a neural PINN and the physical constants are tuned, not known.

### Hybrid with Ruchika's PINN (`pure-pinn` branch) on the real maps

Her per-chip PINN (`pure_pinn/src`, unchanged, copied from branch `pure-pinn`; config `model_b_github.json`) was run on **our** validation and test frames with
**our** sensor readings (`src/eval/real_pinn_runs.py`), so it is scored on exactly the frames and readings the DeepONet sees. It learns nothing across frames. Settings that
differ from her own runs: a shortened budget (600 Adam + 100 L-BFGS steps, about 2 min per solve), one seed, and the sink temperature `T_amb` set to the minimum of the processor's
training maps (real chips have no ambient temperature). Because every solve takes minutes, only a regular sample of frames is used:
**10 validation + 10 test frames for the i5, 20 + 20 for the i7**, at 5, 8 and 10 sensors (180 solves). DeepONet = 5 seeds.

Combining (`src/eval/real_hybrid_pinn.py`, summary `src/eval/real_hybrid_pinn_report.py`; every setting chosen on validation frames only):
**A** blend `w*DeepONet + (1-w)*PINN`; **B** DeepONet plus the heat-equation correction of `src/physics/screened_poisson.py` fitted to its residual at the sensors (no PINN involved);
**C** blend then correction. "The hybrid" in each experiment = the variant with the lowest mean validation error; it was **B** in every experiment.

**Full training data** (test frames, error against temperature rise, mean of 5 seeds; `results/real_pinn_hybrid/`):

| Processor | Sensors | DeepONet | PINN | A: blend | B: + physics correction |
|---|---|---|---|---|---|
| i5 | 5 / 8 / 10 | 0.209 / 0.150 / 0.149 | 0.762 / 0.233 / 0.236 | 0.416 / 0.182 / 0.201 | 0.229 / **0.132** / **0.139** |
| i7 | 5 / 8 / 10 | 0.230 / 0.110 / 0.107 | 0.695 / 0.371 / 0.176 | 0.230 / 0.110 / 0.097 | 0.229 / 0.106 / **0.091** |

With all training data the PINN is much weaker than the DeepONet and blending it in does not help (on the i5 it hurts, because the single validation workload is unlike the test
workload and the blend over-trusts the PINN). The useful hybrid is B; it is better than the DeepONet in 2 of 6 settings and tied in 4 (paired 95% interval over the sampled test frames,
`paired_comparison.csv`), and clearly better than the PINN in all 6.

**Few training workloads (i7)**: the DeepONet is trained on 1, 2 or 4 of the 13 training workloads (nested, seeded; sensor layout kept as fitted on all 13); the PINN is unchanged.
Test error against temperature rise (chart: `results/real_pinn_hybrid/results_few_workloads.png`):

| Sensors | Training workloads | DeepONet | PINN | A: blend | B: + physics correction |
|---|---|---|---|---|---|
| 5 | 2 / 4 / 13 | 0.504 / 0.439 / 0.230 | 0.695 | 0.495 / 0.439 / 0.230 | **0.444** / 0.448 / 0.229 |
| 8 | 2 / 4 / 13 | 0.432 / 0.356 / 0.110 | 0.371 | 0.270 / 0.293 / 0.110 | **0.196 / 0.163** / 0.106 |
| 10 | 2 / 4 / 13 | 0.403 / 0.338 / 0.107 | 0.176 | 0.179 / 0.203 / 0.097 | 0.191 / **0.143** / **0.091** |

With little training data the hybrids are clearly better than the DeepONet (paired intervals above zero in most cells) and, with 8 sensors, better than **both** models by a wide margin
(for example 4 workloads: DeepONet 0.356, PINN 0.371, hybrid 0.163). The blend with the PINN (A) also helps there. Exceptions: with 10 sensors the PINN alone is as good as the hybrid
(2 workloads: tie; with a single workload, the idle recording, the hybrid is worse than the PINN), and with 5 sensors and 4 workloads the hybrid ties the DeepONet. Caveats: the sampled test sets are small
(10 and 20 frames, strongly correlated within a workload) so intervals are optimistic; the i5 has a single validation and a single test workload; the PINN has one seed and a reduced budget;
a single training workload (idle) is an extreme case where the DeepONet fails completely (error 2.5).


### FBPINN and its hybrid with the DeepONet on the real maps (branch `siya-real-fair`)

New per-frame model, scored under the same conditions as Ruchika's PINN: same sampled frames (`subset_idx`, 10 + 10 for the i5, 20 + 20 for
the i7), same noisy readings (`read_sensors`), same metrics (`metrics_row`, non-sensor pixels). The FBPINN (`src/models/real_fbpinn.py`) is her
`train_one_chip` **unchanged** (config `model_b_github.json`, 600 Adam + 100 L-BFGS, one seed, `T_amb` = minimum of the training maps) with only the
field network swapped: 2 x 2 overlapping subdomains, each a tanh MLP (width 32, depth 4), blended by a smooth partition of unity in
`(cos pi x, cos pi y)`, so the adiabatic edges stay exact as in her hard-BC network. 13.2k parameters (hers: 12.8k). Layout fixed in advance, no search.
Hybrids (`src/eval/real_fbpinn_hybrid.py`, report `src/eval/real_fbpinn_report.py`) are the three variants of `real_hybrid_pinn.py` with the FBPINN
in place of the PINN, all settings chosen on validation frames only; DeepONet predictions are the original 5-seed runs of that study (checked: their
per-frame errors reproduce `results/real_pinn_hybrid/per_frame_rise_error*.csv` to 1e-7).

```bash
python -m src.eval.real_fbpinn_runs --processor i5 --shard 0 --shards 6     # 180 solves in total (i5 60, i7 120)
python -m src.eval.real_fbpinn_hybrid                     # and --k-workloads 4, --k-workloads 2
python -m src.eval.real_fbpinn_report
```

Test frames, error against temperature rise (DeepONet and hybrid: mean of 5 seeds; PINN, FBPINN: 1 seed; `results/real_fbpinn/summary.csv`,
paired 95% bootstrap intervals over the sampled test frames in `paired_comparison.csv`, chart `results_full.png`):

| Processor | Sensors | DeepONet | PINN (Ruchika) | **FBPINN** | A: blend (DeepONet, FBPINN) | B: + physics correction (chosen) | gappy POD | train-mean |
|---|---|---|---|---|---|---|---|---|
| i5 | 5 / 8 / 10 | 0.209 / 0.150 / 0.149 | 0.762 / 0.233 / 0.236 | 0.512 / 0.209 / 0.218 | 0.407 / 0.164 / 0.187 | 0.229 / 0.132 / 0.139 | **0.124 / 0.100 / 0.137** | 0.956 |
| i7 | 5 / 8 / 10 | **0.230** / 0.110 / 0.107 | 0.695 / 0.371 / 0.176 | 0.608 / 0.398 / 0.168 | 0.230 / 0.110 / 0.099 | **0.229 / 0.106** / 0.091 | 0.370 / 0.221 / **0.085** | 0.905 |

Few training workloads (i7; the FBPINN uses no training maps, so the same solves are reused):

| Sensors | Training workloads | DeepONet | FBPINN | A: blend | B: + physics correction (chosen) | C: blend + correction |
|---|---|---|---|---|---|---|
| 5 | 2 / 4 | 0.504 / 0.439 | 0.608 | 0.483 / **0.438** | **0.444** / 0.448 | 0.476 / 0.448 |
| 8 | 2 / 4 | 0.432 / 0.356 | 0.398 | 0.294 / 0.307 | **0.196 / 0.163** | 0.252 / 0.198 |
| 10 | 2 / 4 | 0.403 / 0.338 | 0.168 | 0.165 / 0.185 | 0.191 / **0.143** | **0.158** / 0.145 |

How to read it.
* **FBPINN vs Ruchika's PINN** (same frames, budget and pipeline): better in 4 of 6 settings, tie in 2 (i7 with 8 and 10 sensors), never worse. The gain is
  large where the PINN fails (5 sensors: 0.76 -> 0.51 on the i5, 0.70 -> 0.61 on the i7).
* **FBPINN vs DeepONet**: worse in all 6 settings with full training data. With few training workloads it is better at 10 sensors (2 and 4 workloads),
  tied at 8, worse at 5.
* **The hybrid**: by the global rule (lowest mean validation error over all settings) the chosen variant is **B in all three experiments**, and B does
  not use the FBPINN, so the reported hybrid is identical to the existing one (paired difference 0). The FBPINN blend (A) helps where the DeepONet is
  weak, e.g. 8 sensors / 2 workloads 0.432 -> 0.294 and 10 sensors / 2 workloads 0.403 -> 0.165 (better than B there, 0.191), but on validation
  it loses to B on average, so it is not the reported hybrid. We report B as the rule requires and do not pick per setting.
* Gappy POD is still the best method on the i5 (5 and 8 sensors) and ties at 10 sensors on both processors.

Cost (`cost.csv`): DeepONet 119k parameters, ~110 s per trained model; PINN 12.8k, ~110 s per frame; FBPINN 13.2k, ~320 s per frame. The FBPINN
times were measured with 20 single-thread solves running in parallel, the PINN times in the original runs, so the wall-clock numbers are not directly comparable.

Caveats: small sampled test sets (10 i5 / 20 i7 frames, strongly correlated within a workload, so frame-level intervals are optimistic); the i5 has a
single validation and a single test workload; PINN and FBPINN have one seed and a reduced budget; FBPINN was not tuned (one fixed layout). **FNO / PINO
were not run**: the FNO code is in the branch (`src/models/real_fno.py`, `src/train/train_real_fno.py`; input = scattered normalised readings + mask
+ x, y channels) but on this CPU one training step took 2.8 s, so the DeepONet-equal budget (6000 steps x 5 seeds x 3 sensor counts x 4 data
settings) did not fit the compute available, and cutting its budget would have broken the same-compute rule.
