# Sparse-Sensor Thermal Reconstruction: Data-Driven vs. Physics-Informed vs. Hybrid

Comparative study on the semiconductor steady-state thermal dataset
(`../dataset/py_pipeline/dataset`). Reconstruct the full 128x128 temperature
field of a multicore chip from N ∈ {5, 8, 10} sparse point sensors.

| Model | Status | Survey cell ([`PINN_Survey_Updated.pdf`](../PINN_Survey_Updated.pdf), Table II) |
|---|---|---|
| A — data-driven baseline | not implemented yet | operator learning: enforcement `N`, no physical guarantee |
| **B — physics-only PINN** | **implemented** | hard BC (Axis A) x strong form (Axis B); `H`/`S` row |
| C — hybrid | not implemented yet | physics-informed DeepONet: `S`/`S` |

---

## 1. Verified dataset facts

Checked against the generated data, not only the generator README. Several
assumptions in the original study design did **not** hold and the code reflects
the corrected versions:

| Quantity | Reality |
|---|---|
| Splits | already provided: **500 train / 100 val / 100 test** (`thermal_00001`… / `…00501`… / `…00601`…). No re-splitting needed; `data/splits/*.txt` records them. |
| Governing equation | `-k ∇²T = Q` on `[0,1]²`, verified to ~4% FD residual (staircased rectangle edges on a 128 grid) |
| Thermal conductivity `k` | **per-chip**, uniform in `[100, 150]` W/(m·K) — *not* shared across chips |
| Boundary condition | `T = T_boundary` on all of `∂Ω`, **per-chip** in `[20, 40] °C`; equals `min_temperature` to 5e-5 and the matrix border reproduces it exactly. Never zero. |
| Heat sources | axis-aligned **rectangles** `[x0, y0, x1, y1]` with uniform power density (up to 2.5e7 W/m²), 1–10 per chip — *not* Gaussians |
| Matrices | `128 x 128` float32, `M[i, j] = T(x = j/127, y = i/127)` — **row index is y** |
| Sensors | positions land exactly on grid nodes; 5/8/10-sensor files per chip |
| Temperatures | peak from 66 to 15153 °C across chips — per-chip scaling is mandatory |

Run `python tests/test_physics.py` to re-verify all of the above plus the
autodiff Laplacian and the hard-BC construction.

## 2. Normalisation

Model B is solved per chip in non-dimensional units:

```
T_norm = (T - T_boundary) / scale        with scale = max(sensor T) - T_boundary
f_norm = Q / (k * scale)
```

so `-k ∇²T = Q` becomes `∇²T_norm + f_norm = 0` with `T_norm = 0` on `∂Ω`.

`scale` comes from the **sensor readings and metadata only** — never from the
ground-truth field — so no evaluation signal leaks into training. Metrics are
computed after inverting the normalisation, in °C.

## 3. Model B — physics-only PINN

* **Field**: MLP `(x, y) → T̂`, tanh, 4 x 64 by default.
* **Hard BC** (default, `bc_mode: "hard"`):
  `T_norm = d(x, y) · NN(x, y)` with `d = 16 x(1−x) y(1−y)`.
  Exactly `T = T_boundary` on `∂Ω` for every chip, no boundary weight to tune.
  The factor 16 normalises `max d = 1`.
* **Trainable source** (`Q` is unknown), three parametrisations:
  * `gaussian` — isotropic blobs, trainable amplitude/centre/width (study default)
  * `softrect` — smoothed rectangles, **matched to how the data was generated**;
    the matched-model arm of the §10 source-parametrisation ablation
  * `mlp` — free-form second network, softplus output so `Q ≥ 0`
  Amplitudes are exponential (they span orders of magnitude); centres and widths
  pass through bounded sigmoids.
* **Loss**: `L = w_r · L_residual + w_d · L_data` (+ `w_b · L_boundary` in the
  soft ablation only). `weight_scheme: "annealing"` implements gradient-norm
  balancing (Wang et al., graded E1 in the survey) as an alternative to fixed weights.
* **Optimisation**: Adam with cosine decay, collocation resampled every step,
  then L-BFGS with `strong_wolfe` on a *fixed* collocation set (resampling
  breaks its line search). The best iterate by total loss is restored at the end,
  because L-BFGS does step uphill on this objective.

### Boundary ablation arms

`bc_mode` has three settings, which disentangle the two readings of the
secondary experiment in the study design:

| `bc_mode` | `T_boundary` known? | Boundary term | Question it answers |
|---|---|---|---|
| `hard` | yes, by construction | none | headline configuration |
| `soft` | yes, as a penalty target | `L_boundary` at sampled edge points | what does construction buy over a penalty, at equal budget? |
| `free` | **no** | none | can residual + interior sensors alone recover `T_boundary`? |

In `soft`/`free` the scaler offset is 0, so the network must represent the
boundary level itself rather than being handed it through the normalisation.
`boundary_mae` in `metrics.csv` reports the answer for every arm.

## 4. Calibration evidence for the defaults

All on `thermal_00601` (test split, N=5, seed 0, 3000 Adam + 1000 L-BFGS steps).
These are single-chip single-seed numbers used to *pick defaults*, not results.

**Loss balancing** — `w_data=100` is the default: it saturates the sensor fit
without an extreme weight, and gradient annealing bought nothing here, so fixed
weights stay the simpler choice.

| balancing | rel L2 | max abs err | sensor fit (`L_data`) |
|---|---|---|---|
| `w_data=1` | 0.317 | 1292 °C | 2.0e-03 |
| gradient annealing | 0.325 | 1326 °C | 3.7e-06 |
| **`w_data=100`** | **0.295** | 1245 °C | 1.7e-06 |
| `w_data=1e4` | 0.290 | 1217 °C | 3.4e-07 |

**Source parametrisation** — `gaussian` with 6 blobs is the default.

| source | rel L2 | max abs err | source rel L2 | hotspot centre err |
|---|---|---|---|---|
| **gaussian, 6 blobs** | **0.275** | 1191 °C | 0.851 | 0.086 |
| gaussian, oracle count (4) | 0.283 | 1212 °C | 0.854 | 0.077 |
| gaussian, 10 blobs | 0.302 | 1267 °C | 0.865 | 0.134 |
| gaussian 6 + wider field (128x5) | 0.290 | 1239 °C | 0.863 | 0.088 |
| free-form `mlp` source | 0.384 | 1486 °C | 0.911 | — |
| `softrect` | 0.414 | 1553 °C | 0.922 | 0.231 |

Three points worth carrying into the writeup:

1. **The correctly specified source loses.** `softrect` matches how the dataset
   was generated (uniform rectangles) yet does worst: its sharp sigmoid edges
   make the inverse problem much harder to optimise, so the mis-specified but
   smooth Gaussian wins. Model-form correctness and trainability pull in
   opposite directions here.
2. **The oracle hotspot count does not help.** Six free blobs match four correct
   ones, so the main comparison needs no oracle-count variant.
3. **Localisation is good while source *shape* recovery is poor**
   (`source_rel_l2` ~0.85 against `source_centre_err` ~0.08). Model B finds
   *where* the heat is, not *how it is distributed* — the right claim to make
   given the engineering interest is hotspot location.

**The oracle-source floor** (`tests/test_forward_solve.py`, same chip): with the
*true* source handed to the network and no sensor data at all, rel L2 = **0.086**,
max abs error 218 °C, peak location error 0.000. So Model B's ~0.28 is dominated
by source under-determination from sparse sensors, not by the network's ability
to represent the field. Quote this floor whenever Model B is compared with A or C:
it separates the inverse-problem limit from the representation limit.

Note that `thermal_00601` is a *hard* chip: 4 overlapping hotspots in a 2651 °C
peak field, and Model B scores ~0.28 on it against ~0.13 averaged over the chips
in Sec. 7.
The calibration numbers above are therefore comparable to each other but are not
representative of Model B's typical accuracy.

## 5. Layout

```
src/
  data_loader.py   metadata/sensor/matrix loading, per-chip + global scalers, splits
  losses.py        autodiff Laplacian, PDE residual, data/boundary losses,
                   collocation sampling, gradient-norm annealing  (shared with Model C)
  models/pinn.py   hard-BC temperature field + the three source parametrisations
  train_pinn.py    Model B driver: one solve per (chip, sensor count, seed)
  evaluate.py      metrics, results/metrics.csv, comparison table, error-vs-N plot
tests/
  test_physics.py       8 checks: autodiff Laplacian, manufactured solution,
                        exact hard BC, dataset/PDE consistency, indexing
  test_forward_solve.py oracle-source forward solve -> representation floor
experiments/
  configs/         one JSON per configuration
  run_model_b.sh   parallel matrix runner (per-process CSVs, merged afterwards)
results/
  metrics.csv  parts/  runs/  logs/  figures/
```

## 6. Usage

```bash
# one-time: record the split ID lists
python -c "import sys;sys.path.insert(0,'src');import data_loader as d;print(d.write_split_lists())"

# sanity-check on a single chip (milestone 3)
python src/train_pinn.py --split test --chips thermal_00601 --sensors 5 --seeds 0 --save-figs

# Model B arm of the experiment matrix, test set subsampled for cost (§10).
# 4 concurrent solves, per-process CSVs under results/parts/
experiments/run_model_b.sh 20 5,8,10 0,1,2 4
python src/evaluate.py --merge results/parts --plot

# boundary ablation arms
OVERRIDE='{"model":{"bc_mode":"soft"}}' TAG=soft PARTS=results/parts_bc \
  experiments/run_model_b.sh 3 5 0,1,2 3
OVERRIDE='{"model":{"bc_mode":"free"}}' TAG=free PARTS=results/parts_bc \
  experiments/run_model_b.sh 3 5 0,1,2 3

# representation floor for a given chip
python tests/test_forward_solve.py --chip thermal_00601

# ablations
python src/train_pinn.py --chips thermal_00601 --override '{"model":{"source":{"kind":"softrect"}}}'
python src/train_pinn.py --chips thermal_00601 --override '{"model":{"bc_mode":"free"}}'

# summarise
python src/evaluate.py --metric rel_l2
```

`--n-chips K` takes a *deterministic* subsample of the split (fixed RNG), so the
Model B arm is reproducible and the writeup can state exactly which chips were
used. `--override` merges a JSON dict into the config, which is how the ablation
arms are run without duplicating config files.

### Notes and caveats

* **Cost asymmetry is a result, not a nuisance.** Model B re-solves per chip;
  `train_seconds` / `inference_seconds` are recorded per run so the comparison
  table can report it (§6 of the design).
* **`n_blobs`** defaults to 6. Setting it to `null` or `"oracle"` uses the chip's
  true `hotspot_count` from metadata — label this an *oracle-count* variant in
  any writeup, since a deployed solver would not know M. With N=5 sensors and
  6 blobs the source is heavily under-determined; that is the point of the
  physics term, and `source_rel_l2` / `source_centre_err` measure how much of
  the true source is actually recovered.
* Ground-truth fields, `hotspot_positions` and `hotspot_power_values` are read
  **only** inside `evaluate.py` / `evaluate_source`, never in the training path.

## 7. Results so far (Model B only)

Real runs, `results/metrics.csv`: 3 test chips (`thermal_00652`, `00664`, `00684`)
x N in {5, 8, 10} x 2 seeds = 18 solves, default config. `+/-` is **seed** std
(per-chip runs averaged within a seed first); chip-to-chip spread is much larger
at ~0.04-0.05.

| N sensors | rel L2 | max abs err | peak loc err | boundary MAE | sensor RMSE | s/chip |
|---|---|---|---|---|---|---|
| 5 | 0.135 +/- 0.002 | 558 °C | 0.075 | **0.0** | 4.0 °C | 873 |
| 8 | 0.124 +/- 0.004 | 430 °C | 0.064 | **0.0** | 2.8 °C | 865 |
| 10 | 0.126 +/- 0.002 | 478 °C | 0.060 | **0.0** | 4.9 °C | 710 |

Findings to carry into the writeup:

* **Sensor count barely matters for Model B.** 5 -> 10 sensors moves rel L2 from
  0.135 to 0.126, less than the chip-to-chip spread. The physics term, not the
  data, is doing the work. Expect the opposite for Model A, which is the point
  of the §5 ablation plot.
* **`boundary_mae` is exactly 0.0 on every run**, with no boundary weight in the
  loss. This is the Axis-A hard-enforcement claim demonstrated, not asserted.
* **The peak is systematically under-predicted** while its *location* is nearly
  exact (0.008 on `thermal_00602`, under one pixel). The smooth Gaussian source
  flattens the true rectangular heater's plateau. So `max_abs_err` carries
  information `rel_l2` hides -- keep both in the headline table.
* **Cost asymmetry**: ~12-15 min *per chip*, re-solved from scratch every time,
  against a single training run plus millisecond inference for Models A and C.
  Wall-clock figures above were measured at 4-way concurrency on a 12-core
  CPU-only box; state the concurrency setting whenever quoting them.

Example figures in `results/figures/`:
`error_vs_sensors.png` (headline plot) and
`thermal_0060{1,2}_N5_seed0_hard_gaussian_figdemo.png` (true / predicted / error triptychs).

### Not done yet

* Models A and C (`losses.py` is written so Model C imports the physics unchanged).
* Full-scale arm: 20 chips x 3 N x 3 seeds via `experiments/run_model_b.sh 20 5,8,10 0,1,2 4` (~3 h here).
* The `soft` / `free` boundary ablation arms -- implemented and wired, not yet run.
