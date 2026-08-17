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

## 4. Layout

```
src/
  data_loader.py   metadata/sensor/matrix loading, per-chip + global scalers, splits
  losses.py        autodiff Laplacian, PDE residual, data/boundary losses,
                   collocation sampling, gradient-norm annealing  (shared with Model C)
  models/pinn.py   hard-BC temperature field + the three source parametrisations
  train_pinn.py    Model B driver: one solve per (chip, sensor count, seed)
  evaluate.py      metrics, results/metrics.csv, comparison table
tests/test_physics.py
experiments/configs/
results/metrics.csv, results/runs/, results/figures/
```

## 5. Usage

```bash
# one-time: record the split ID lists
python -c "import sys;sys.path.insert(0,'src');import data_loader as d;print(d.write_split_lists())"

# sanity-check on a single chip (milestone 3)
python src/train_pinn.py --split test --chips thermal_00601 --sensors 5 --seeds 0 --save-figs

# Model B arm of the experiment matrix, test set subsampled for cost (§10)
python src/train_pinn.py --split test --n-chips 20 --sensors 5,8,10 --seeds 0,1,2 --quiet

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
