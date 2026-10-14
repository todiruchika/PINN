# Per-chip physics-only PINN (Model B)

Reconstruct a chip's full 128x128 temperature map from N ∈ {5, 8, 10} sparse
noisy sensors, using **only the governing equation and that chip's own sensor
readings**. One small network is fitted **from scratch for every chip**; nothing
is learned across chips, and the ground-truth field is used only to score the
result.

This is the pure-PINN reference arm. Models A (data-driven) and C (hybrid) are
not implemented here.

> **Dataset.** This code targets the HotSpot-style dataset on `main`
> (`data/raw/dataset`, 700 chips = 500 train / 100 validation / 100 test, POD-QR
> sensor files in `data/generated`). An earlier version targeted a different,
> Poisson-type dataset; its results are kept in `results_legacy_old_dataset/` and
> are **not comparable** with anything produced now.

---

## 1. What is solved

The dataset's PDE (`generator/pde_solver.py`, `PROTOCOL.md`):

```
-div( t_die · k(T) · grad T ) + h_eff · (T - T_amb) = q(x, y, T)        on the 12 mm die
dT/dn = 0                                                              on all four edges
k(T)  = 22714 / (T[K] - 99) + 2   clipped to [20, 160] W/(m·K)
q     = (p_dyn + p_leak0 · exp(β (T - 25))) · 1e4   W/m², block-wise uniform
```

Per chip the code knows `T_amb` (metadata) and the sensor readings. It does
**not** know the block layout or powers: the source `q` is learned.

Normalised units (see `src/losses.py`): `θ = (T - T_amb)/S`, `S` = hottest sensor's
rise above ambient, coordinates scaled to the unit square. The equation becomes

```
-ε · div( k(T) grad θ ) + θ = q̃ ,      ε = t_die / (h_eff L²) ≈ 5.6e-4 ,   dθ/dn = 0
```

### Model

| Part | Choice |
|---|---|
| Field `θ(x, y)` | tanh MLP, 4 × 64 by default |
| Boundary | **hard** (default): `θ = NN(cos πx, cos πy)` has `dθ/dn = 0` on every edge for *any* weights, so there is no boundary weight to tune. `soft`: plain MLP + Neumann penalty. `free`: plain MLP, no boundary term. |
| Source | learned *dynamic* power density; leakage `1 + f·exp(β(T - 25))` is applied on top. `gaussian` (default, 6 blobs), `softrect`, or `mlp`. |
| Loss | `w_residual · mean(R²) + w_data · mean((θ_sensor - θ̂)²)` (+ Neumann term in `soft`) |
| Optimiser | Adam with cosine decay, then L-BFGS on a fixed collocation set; best iterate restored |

### Assumptions to be aware of

These are inherited from `PROTOCOL.md` ("ASSUMPTION — NOT SOURCED") and from the
generator, and live in the `physics` block of the config:

* `t_die = 500 µm`, `h_eff = 6256.3 W/(m²·K)`, `β = 0.012 /K`, `T0 = 25 °C`.
* `leak_fraction = 0.0425`: the generator gives each block type its own leakage
  fraction (CPU/GPU 0.05, Cache 0.04, MC 0.03). A physics-only model does not
  know block types, so it uses their mean. Set it to `0` to drop leakage.

---

## 2. Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # numpy pandas scipy pyyaml matplotlib pillow pytest torch
```

CPU is enough (`"device": "cpu"` in the config). **Data needed** (both committed):

```
data/raw/dataset/<split>/matrices/thermal_NNNNN.npy      # ground-truth T   (evaluation only)
data/raw/dataset/<split>/matrices/thermal_NNNNN_Q.npy    # ground-truth q   (evaluation only)
data/raw/dataset/<split>/metadata.csv
data/generated/<split>/thermal_NNNNN_sensors_{5,8,10}.json   # POD-QR sensor readings
```

Sanity-check the data and the physics before training:

```bash
python -m src.data.validate_dataset --dir data/raw/dataset      # from the repo root
python pure_pinn/tests/test_physics.py
```

All paths inside `pure_pinn/` are resolved from the file location, so the
commands below work from any directory. They are written for the repo root.

---

## 3. Train one chip (manual run)

```bash
python pure_pinn/src/train_pinn.py \
    --split train --chips thermal_00001 --sensors 5 --seeds 0
```

That fits one PINN for chip `thermal_00001` with 5 sensors and seed 0, prints
progress, and appends **one row** to `pure_pinn/results/metrics.csv`. At the
default budget (5000 Adam + 2000 L-BFGS steps) it took **~2m50s** on a 4-core CPU
running alone, and 3-5.5 minutes per solve with two solves running side by side at
2 threads each.

Pick the chip, split, and sensor count yourself:

| Goal | Command |
|---|---|
| Another chip | `--chips thermal_00002` (the id must belong to `--split`) |
| Another split | `--split validation` or `--split test` (`val` also works) |
| 8 or 10 sensors | `--sensors 8` / `--sensors 10` |
| Several sensor counts | `--sensors 5,8,10` |
| Several seeds | `--seeds 0,1,2` |
| Several chips | `--chips thermal_00001,thermal_00002` |
| K chips of a split (fixed-seed random subsample, same K chips every time) | `--n-chips 20` |
| Every chip in a split | leave out `--chips` and `--n-chips` |

Which split a chip belongs to: the splits are interleaved across ids 1–700, so check
`data/raw/dataset/splits.csv` (e.g. `thermal_00001` → train, `thermal_00008` →
validation, `thermal_00014` → test). A chip id given with the wrong `--split` is
rejected with an error.

### Useful flags

| Flag | Meaning |
|---|---|
| `--save-figs` | write a true / predicted / error picture to `pure_pinn/results/figures/` |
| `--save-fields` | save the predicted 128×128 field as `.npy` in `pure_pinn/results/runs/` |
| `--quiet` | print only the final line per solve |
| `--metrics PATH` | write the metrics row to a different CSV |
| `--runs-dir DIR` | where per-run JSON (config, loss history, scaler) goes |
| `--tag NAME` | suffix for the run files, to tell variants apart |
| `--config FILE` | a different config JSON (default `experiments/configs/model_b_default.json`) |
| `--override '{...}'` | JSON merged on top of the config |

Example with figure and field saved:

```bash
python pure_pinn/src/train_pinn.py --split train --chips thermal_00001 \
    --sensors 5,8,10 --seeds 0 --save-figs --save-fields
```

### Variants via `--override`

```bash
# shorter budget (quick check, ~10 s)
--override '{"training":{"adam_steps":300,"lbfgs_steps":50}}'

# boundary-treatment ablations (soft penalty / no boundary term)
--override '{"model":{"bc_mode":"soft"}}'
--override '{"model":{"bc_mode":"free"}}'

# other source parametrisations
--override '{"model":{"source":{"kind":"softrect"}}}'
--override '{"model":{"source":{"kind":"mlp"}}}'

# number of source blobs (or "oracle" = use the chip's true block count; label it an oracle run)
--override '{"model":{"source":{"n_blobs":4}}}'

# drop the temperature-dependent leakage
--override '{"physics":{"leak_fraction":0.0}}'
```

Use `--tag` together with an override so the variant's run files do not overwrite
the default ones, and `--metrics` to keep its rows in a separate CSV.

### What you get

* `results/metrics.csv`: one row per solve (appended; running the same solve twice
  adds two rows). Columns:

  | Group | Columns |
  |---|---|
  | Run | `model, bc_mode, source_kind, split, sample_id, n_sensors, seed` |
  | Field error (non-sensor nodes, °C) | `rel_l2, rmse, mae, mse, max_abs_err` |
  | Peak | `peak_temp_err, hotspot_loc_cells, pred_peak_temp, true_peak_temp` |
  | Constraint / fit | `boundary_flux_mean` (mean \|dT/dn\| on the edges, K/m), `sensor_rmse` |
  | Source recovery | `source_rel_l2, source_centre_err` |
  | Cost | `n_params, train_seconds, inference_seconds, final_loss, final_residual, final_data_loss` |

  All field errors are over the **non-sensor nodes only**, as `PROTOCOL.md` requires.
* `results/runs/<chip>_N<n>_seed<s>_<bc>_<source>[_tag].json`: config, loss history, scaler.
* `*_pred.npy` and `results/figures/*.png` when requested.

---

## 4. Many chips in parallel

`experiments/run_model_b.sh` runs one process per (chip, sensor count, seed), a few
at a time, each writing its own CSV (concurrent appends to one file are not safe):

```bash
cd pure_pinn
# usage: experiments/run_model_b.sh [N_CHIPS] [SENSORS] [SEEDS] [JOBS] [CONFIG]
SPLIT=test experiments/run_model_b.sh 20 5,8,10 0,1,2 4

# explicit chip list instead of a subsample
SPLIT=train CHIPS=thermal_00001,thermal_00002 experiments/run_model_b.sh 2 5,8,10 0 2

# a variant arm, kept apart
OVERRIDE='{"model":{"bc_mode":"soft"}}' TAG=soft PARTS=results/parts_bc \
    SPLIT=test experiments/run_model_b.sh 3 5 0,1,2 3

# merge the per-process CSVs, print the comparison table, plot error vs sensors
python src/evaluate.py --merge results/parts --plot
```

Environment knobs: `SPLIT` (default `test`), `CHIPS`, `OVERRIDE`, `TAG`, `PARTS`,
`THREADS_PER_JOB` (default 2), `FIGS=0` to skip figures. Keep `JOBS × THREADS_PER_JOB`
near your core count. Wall-clock times are only comparable at the same `JOBS`.

Cost to plan around: one solve is ~3-5 minutes, so all **500 train chips × 3 sensor counts
× 1 seed** is 1,500 solves, i.e. roughly **75-120 hours** of compute on a 4-core box.
Running jobs side by side does not speed a solve up, so measure before launching
anything large.
Use `--n-chips` to subsample. There is no shared model, so nothing is saved for
re-use: the 500 train chips are simply a pool of chips to tune on or evaluate on.

---

## 5. Tests and diagnostics

```bash
python pure_pinn/tests/test_physics.py                      # 18 checks, a few seconds
python pure_pinn/tests/test_forward_solve.py --chip thermal_00001 --split train
```

* `test_physics.py` checks the autodiff operators against analytic results, that
  `k(T)` equals the generator's `k_silicon`, that the hard BC is exactly adiabatic,
  that the **dataset's own fields satisfy the normalised PDE** (median 7% residual,
  while a wrong `t_die` gives >100%), that stored `Q` integrates to the recorded
  input power, sensor/matrix index alignment, and that metrics ignore sensor nodes.
* `test_forward_solve.py` is the **oracle floor**: it gives the network the true source
  and no sensor data. It reached **rel L2 = 0.0012** on `thermal_00001`, so the PDE,
  units, boundary embedding and loss are not what limits Model B. Quote this whenever
  Model B is compared with another model.

---

## 6. Verification run on the new dataset

Not a result set, just evidence that the ported model works end to end: 4 **test** chips
(`thermal_00014, 00022, 00025, 00039`), N ∈ {5, 10}, seed 0, default config. Rows are in
`results/verification/metrics_4_test_chips.csv`. Errors are on non-sensor nodes, in °C.

| chip | N | rel L2 | MAE | max abs | peak err | hotspot loc (cells) | sensor RMSE |
|---|---|---|---|---|---|---|---|
| 00014 | 5 | 0.071 | 1.94 | 7.4 | 0.2 | 32.2 | 0.02 |
| 00014 | 10 | 0.071 | 1.96 | 6.8 | 4.1 | 14.9 | 0.07 |
| 00022 | 5 | 0.216 | 8.92 | 40.8 | 12.1 | 44.1 | 0.00 |
| 00022 | 10 | 0.081 | 3.04 | 24.8 | 2.2 | 16.3 | 0.05 |
| 00025 | 5 | 0.119 | 2.79 | 22.7 | 21.6 | 17.9 | 0.00 |
| 00025 | 10 | 0.060 | 1.62 | 11.5 | 9.8 | 12.0 | 0.04 |
| 00039 | 5 | 0.096 | 2.55 | 23.0 | 9.3 | 26.6 | 0.02 |
| 00039 | 10 | 0.095 | 2.57 | 23.4 | 11.1 | 44.7 | 0.08 |
| **mean ± std over chips** | 5 | 0.125 ± 0.064 | | | 10.8 ± 8.8 | 30.2 ± 11.0 | |
| **mean ± std over chips** | 10 | 0.077 ± 0.015 | | | 6.8 ± 4.3 | 22.0 ± 15.2 | |

For rough context, a gappy-POD reconstruction (the saved POD-QR basis, scaled from the
same sensor readings; **my quick adaptation, not the repo's baseline entry point**) gave
mean rel L2 ≈ 0.23 at N=5 and ≈ 0.09 at N=10 on the same four chips. So Model B looks
better at 5 sensors and comparable at 10, but that is 4 chips and 1 seed with a large
chip-to-chip spread: do not quote it as a result. A proper comparison needs the
protocol's 5 seeds and many more chips.

What the table says about the model:

* The field error is **moderate and the hotspot is poorly placed**. Hotspot localisation
  is 12-45 cells (1 cell = 0.094 mm, so 1-4 mm) and the peak temperature is off by
  several °C. Sensor RMSE is near zero, so the model fits its own data; it just does not
  know where the heat sources are.
* That is the **under-determined inverse problem**: with 5-10 sensors many source layouts
  fit the readings and satisfy the PDE. It is not an implementation fault: the oracle
  test (section 5) reproduces the true field to rel L2 0.0012 once the source is given.
* On `thermal_00001` (N=5) a 300-step run scored rel L2 0.043 while the full 7,000-step
  run scored 0.080, with the peak over-predicted (63 °C against a true 42.6 °C). That is
  one chip, but it suggests the later iterations lock in a spurious hotspot, so test a
  shorter budget (or early stopping on held-out sensors) before assuming the default
  budget is best.

Judge Model B by `rel_l2`, `peak_temp_err` and `hotspot_loc_cells` together, over many
chips and seeds, never from one chip.

**Error against sensor count, 3 seeds** (`results/verification/metrics_chips_1_2_3_seeds_0_2.csv`: chips
`thermal_00001-00003`, N ∈ {5, 8, 10}, seeds 0-2, 27 solves). Relative L2, mean over the three chips,
then mean ± std over seeds: N=5 0.109 ± 0.003, N=8 0.090 ± 0.002, N=10 0.073 ± 0.002. N=10 beat N=5 in
all 9 (chip, seed) pairs, so the gain from more sensors is not seed noise. It is not monotonic per chip
(e.g. `thermal_00003` is worse at N=8 than at N=5, reproducibly), and hotspot location does not improve
with N (about 28-30 cells throughout). Only 3 chips, all from the train split.

---

## 7. Layout

```
src/
  data_loader.py   new-dataset loading, per-chip scaler (ambient offset)
  losses.py        Physics constants, flux-divergence PDE residual, data/Neumann losses
  models/pinn.py   hard-Neumann temperature field + source parametrisations
  train_pinn.py    driver: one solve per (chip, sensor count, seed)
  evaluate.py      non-sensor-node metrics, results CSV, merge/plot
  make_flowchart.py  design-flow figure (unchanged)
tests/             test_physics.py, test_forward_solve.py
experiments/       configs/model_b_default.json, run_model_b.sh
results/verification/  the 8-solve check above
results_legacy_old_dataset/   results and split lists from the previous (incompatible) dataset
docs/              original implementation note (describes the previous dataset)
```
