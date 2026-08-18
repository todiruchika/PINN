# Sparse-Sensor Thermal Reconstruction: Data-Driven vs. Physics-Informed vs. Hybrid

Comparative study on the semiconductor steady-state thermal dataset
(`../dataset/py_pipeline/dataset`). Reconstruct the full 128x128 temperature
field of a multicore chip from N ∈ {5, 8, 10} sparse point sensors.

| Model | Status | Survey cell ([`PINN_Survey_Updated.pdf`](../PINN_Survey_Updated.pdf), Table II) |
|---|---|---|
| **A — data-driven DeepONet baseline** | **implemented (this branch)** | operator learning: no physical guarantee |
| B — physics-only PINN | in progress on another branch | strong-form PINN, hard Dirichlet BC |
| C — hybrid (physics-informed DeepONet) | not started | physics-informed DeepONet |

This branch implements **Model A only** — the pure data-driven baseline, using
the DeepONet option from Sec. 3.1 (not the CNN/U-Net alternative). Models B
and C reuse the same shared infrastructure (`data_loader.py`, `evaluate.py`,
`utils.py`) and are expected to land on sibling branches before the final
merge.

---

## 1. Verified dataset facts

Checked against the generated data, not only the generator README — several
assumptions in the original study design did **not** hold, and the code
reflects the corrected versions:

| Quantity | Reality |
|---|---|
| Splits | already provided: **500 train / 100 val / 100 test** (`thermal_00001`… / `…00501`… / `…00601`…). No re-splitting needed; `data/splits/*.txt` records them. |
| Governing equation | `-k ∇²T = Q` on `[0,1]²` |
| Thermal conductivity `k` | **per-chip**, uniform in `[100, 150]` W/(m·K) — *not* shared across chips |
| Boundary condition | `T = T_boundary` on all of `∂Ω`, **per-chip** in `[20, 40] °C`; equals `min_temperature` and the matrix border reproduces it exactly. **Never zero.** |
| Heat sources | axis-aligned **rectangles** `[x0, y0, x1, y1]` with uniform power density (up to 2.5e7 W/m²), 1–10 per chip |
| Matrices | `128 x 128` float32, `M[i, j] = T(x = j/127, y = i/127)` — row index is *y* |
| Sensors | positions land exactly on grid nodes; separate 5/8/10-sensor JSON files per chip |
| Temperatures | peak from 66 to ~15,153 °C across chips — dataset-wide scaling has to span that whole range for a cross-chip model |

None of this changes Model A's design (it is agnostic to `k` and the BC value
by construction — it never sees the PDE), but it matters for the shared
normalisation code and for Models B/C.

## 2. Model A — DeepONet baseline

* **Branch net**: MLP, input = that chip's N sensors flattened as
  `[x_1, y_1, T_1, ..., x_N, y_N, T_N]` (positions *and* readings — sensor
  placement is randomised per chip, so the network needs to know where each
  reading came from). Output: `p`-dim embedding.
* **Trunk net**: MLP, input = query coordinate `(x, y)`. Output: `p`-dim
  embedding. Optional random Fourier positional features are wired in
  (`model.net.fourier_freqs`) but off by default — an ablation run found they
  made no measurable difference on this dataset, so the simpler network is
  the default.
* **Head**: `T_hat(x, y) = branch(sensors) . trunk(x, y) + bias` — the
  standard DeepONet combination.
* **No physics anywhere**: no PDE residual, no boundary term, no per-chip
  fine-tuning. This is deliberately the weak baseline in the comparison.
* **Normalisation**: `GlobalScaler` (in `data_loader.py`) — dataset-wide
  min-max fit from the *train* split's `min_temperature` / `peak_temperature`
  columns only. Positions are already in `[0, 1]` and need no scaling.
  Predictions are denormalised before any metric is computed.
* **Training protocol**: trained **once** per `(n_sensors, seed)` on the full
  500-chip train split (mini-batches of chips, a fresh random subset of query
  grid points sampled every step), early-stopped on val relative-L2, then
  evaluated on every one of the 100 held-out test chips — the "train once,
  generalise" arm of the study, as opposed to Model B's per-chip solve.

### A note on the result

With only 5–10 point sensors and no physical prior, a purely data-driven map
has to infer the location, count (1–10), and power (spanning ~2 orders of
magnitude) of rectangular hotspots from a handful of scalars. Relative-L2 in
the 40–55% range at this sensor density is the expected weak-baseline
outcome, not a bug — reducing it further would need either far more training
chips, per-chip fine-tuning (which stops it from being Model A), or the
physics prior that Models B/C add. This is exactly the gap the comparison is
meant to surface (Sec. 1, Table II: "operator learning, no physical
guarantee").

## 3. Layout

```
src/
  data_loader.py          metadata/sensor/matrix loading, per-chip + global scalers, splits (shared)
  evaluate.py              metrics, results/metrics.csv, comparison table (shared)
  utils.py                 seeding, timing, config loading (shared)
  models/
    deeponet_baseline.py   Model A network: branch/trunk MLPs + optional Fourier trunk features
  train_baseline.py        Model A driver: train once per (sensor count, seed), eval on every test chip
tests/test_deeponet.py     shape checks, verified-facts checks, tiny overfit sanity check
experiments/configs/model_a_default.json
results/metrics.csv, results/checkpoints/, results/figures/
```

## 4. Usage

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # or source .venv/bin/activate on Linux/Mac

# sanity checks (shapes, verified dataset facts, tiny overfit test)
python tests/test_deeponet.py

# smoke test on the small bundled sample (20/5/5 chips)
python src/train_baseline.py --data-root ../dataset/py_pipeline/my_dataset \
    --sensors 5 --seeds 0 --epochs 20 --metrics-out results/_smoke.csv

# full Model A arm of the experiment matrix (9 runs: N in {5,8,10} x seed in {0,1,2})
python src/train_baseline.py --config experiments/configs/model_a_default.json \
    --sensors 5,8,10 --seeds 0,1,2 --save-checkpoints

# summarise into the comparison table + headline figure
python src/evaluate.py --metric rel_l2 --plot
```

`train_baseline.py` writes one row per `(model, n_sensors, seed, test chip)`
to `results/metrics.csv` via the shared `evaluate.append_metrics_row`, so its
output composes directly with Model B/C runs in the same file — the table in
`evaluate.py`'s `comparison_table()` already groups by `model` and averages
seeds within a chip first, exactly as Sec. 6/7 of the design doc specifies.

### Notes and caveats

* Model A's `bc_mode` / `source_kind` / `source_rel_l2` / `final_residual`
  metric columns are always blank — they're PINN-specific (Model B/C) and the
  shared `evaluate.py` schema leaves them empty for rows that don't apply.
* `train_seconds` is the *whole* training run's cost, repeated on every test
  chip's row (since Model A trains once, not per chip) — this is what makes
  the cost-asymmetry comparison against Model B (which pays per chip) correct
  when averaged.
* Checkpoints go to `results/checkpoints/` and are git-ignored; rerun
  `train_baseline.py` to reproduce them rather than committing multi-MB
  weight files.
