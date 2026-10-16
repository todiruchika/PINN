# FBPINN and XPINN: results and comparison

Branch: `siya-fbpinn` (built on `siya-fno-pino`, which holds the FNO/PINO work and the shared data loading,
metrics and physics check). Code: `src/models/fbpinn.py`, `src/models/xpinn.py`, `src/models/pod_prior.py`,
`src/train/fbpinn.py` (also trains XPINN), `src/train/xpinn.py`.
Tables and figures here are built from the saved `results.json` files by `src/eval/compare_models.py`,
`src/eval/scarcity_table.py` and `src/eval/physics_consistency.py`. The FNO/PINO write-up is in `results/thesis/RESULTS.md`.

## 1. The two models

Both models reconstruct a chip's 128×128 temperature map from 5, 8 or 10 noisy sensors. A shared encoder turns the sensor
readings (positions, readings, ambient) into a summary vector that every sub-network receives. Outputs are the temperature
rise θ and the heat source q̃ (as in the FNO model). Only θ is scored.

- **FBPINN (finite basis PINN):** the die is covered by overlapping windows (9, or 16 in one run), one small network per window.
  Outputs are blended by smooth window weights that sum to one, so the field is continuous with no matching conditions.
- **XPINN (extended PINN):** the die is split into 3×3 non-overlapping tiles, one small network per tile. A loss at each shared
  edge makes neighbouring tiles agree on temperature, source and temperature gradient.

Both use about 24,000 parameters, against 118,000 for the DeepONet and 533,000 for the FNO.

## 2. Protocol

Same data (500 train / 100 validation / 100 test), same sensor files, same score as the DeepONet branch
(`src.eval.metrics.all_metrics` on non-sensor nodes, checked in `src/train/train_deeponet.py`), best-validation checkpoint,
test scored once. Physics weight 0.01 unless stated, source-loss weight 0.1.

**Efficient training mode (used for the "fast" runs).** The first FBPINN runs pushed the full 128×128 map of 8 chips through the model every step.
The fast mode instead uses 32 chips per step, 1024 random grid nodes for the data losses, and the full-grid physics check on only 2 of the 32 chips.
That gives about 4× more chips per step at lower cost per step. A 6000-step run sees 192,000 chip samples (DeepONet: about 200,000).

## 3. Main results (test set, relative L2 error, lower is better)

| Model | 5 sensors | 8 sensors | 10 sensors |
|---|---|---|---|
| FNO/PINO (final) | 13.20% (1 seed) | 10.79% (1) | 9.78% ± 0.20 (3) |
| FNO/PINO + POD estimate | 11.23% (1) | n/a | n/a |
| FBPINN, full-grid, 3000 steps × 8 chips | 12.25% (1) | 10.02% (1) | 8.11% (1) |
| FBPINN fast | 11.35% (1, 10k steps) | 8.95% ± 0.05 (3) | 6.76% ± 0.03 (3) |
| FBPINN fast, 16 windows | n/a | n/a | **6.25%** (1) |
| XPINN, physics on, tile edges masked | 11.43% (1) | 8.72% (1) | 6.45% ± 0.03 (3) |
| XPINN, patch-averaged physics, tile edges masked | n/a | n/a | 6.37% (1) |
| XPINN, no physics | 11.38% (1) | 8.63% (1) | 6.40% (1) |
| DeepONet (teammates, 5 seeds, 400 epochs) | 11.46% | 8.62% | 6.13% |

Full table with MAE, peak error, hotspot error, steps, batch and parameter counts: `summary.md`. Figure: `error_vs_sensors.png`.

- The fast decomposition models close most of the gap to the DeepONet. At 5 sensors both match it; at 8 sensors XPINN matches it and FBPINN is
  about 0.3 points behind. At 10 sensors the best single-seed result (FBPINN, 16 windows, 6.25%) is 0.12 points behind the DeepONet's 5-seed
  mean; the best multi-seed XPINN is 0.3 behind.
- Both the architecture and the efficient training mode contribute to the gain over FNO/PINO. The full-grid FBPINN (12.25 / 10.02 / 8.11%) already
  beats the FNO (13.20 / 10.79 / 9.78%), and the fast mode adds a further 0.9 to 1.4 points (11.35 / 8.95 / 6.76%).
- The 16-window FBPINN (40,000 parameters) improves on the 9-window one (6.76% → 6.25%, single seed).
- At 5 sensors every model lands between 11.2% and 11.5%, including the DeepONet. That sensor count looks limited by the information in the
  sensors, not by the model.
- A larger sensor encoder (256 wide, 120k parameters) made no difference (6.80% vs 6.76%).

## 4. Does physics help? An honest answer: not measurably, here

| Comparison (10 sensors, 500 training chips) | Without physics | With physics |
|---|---|---|
| FBPINN | 6.83% (1 seed) | 6.76% ± 0.03 (3 seeds) |
| XPINN | 6.40% (1 seed) | 6.45% ± 0.03 (masked, 3 seeds); 6.37% (patch-averaged, 1 seed) |

Data-scarcity test (XPINN, 10 sensors, seed 0, trained on only some of the chips):

| Physics term | 25 chips | 50 chips | 500 chips |
|---|---|---|---|
| none | 10.66% | 8.97% | 6.40% |
| pointwise (tile edges masked) | 10.54% | 9.11% | 6.45% |
| 16×16 patch-averaged (tile edges masked) | 10.62% | 8.98% | 6.37% |

All differences are 0.15 points or less, about the size of the seed spread (±0.03 at 500 chips, unknown at 25 and 50 chips because each is one seed).
**In these experiments physics neither helps nor hurts once it is applied correctly.**

### 4.1 Where physics did hurt, and why

- **XPINN with plain physics stalled near 16–17%.** Output jumps at the hard tile edges produce a huge residual in the grid physics check, and
  the optimiser flattened the field to remove it. Leaving the nodes next to tile edges out of the physics loss (`--seam-mask`) fixed it
  (16.2% → 6.45%). Three other fixes did not help: wider tile networks (96 hidden units, 16.11% test error, saved in `summary.md`), smaller interface
  weights (0.01 and 0.1) and a small physics weight (0.001). The last three were stopped early at about 16.5–17.1% validation error, so they have no saved final results.
- **FNO/PINO got worse with a physics weight of 0.1** (13.64% vs 10.40% at 3000 steps).

### 4.2 Why physics adds little: the true maps do not satisfy the grid equation

`python -m src.eval.physics_consistency` passes the ground-truth maps through the grid version of the heat equation (40 validation chips):

| Measure | Value |
|---|---|
| Pointwise residual as a fraction of the signal | 32.9% |
| Share of the squared residual in the worst 5% of pixels | 99% |
| Residual away from source-block edges | 7.0% |
| Residual averaged over 4×4 / 8×8 / 16×16 patches | 17.8% / 12.2% / 7.9% |

If the grid equation matched the dataset, these would be near zero. A fit of the equation's scale factors gave about 0.92 for diffusion and 0.93
for the source, so a wrong coefficient is not the cause. The mismatch is concentrated at hotspots and sharp power-block edges, which
is where accuracy matters most. This fits a data-generation effect: the fields were solved on a finer mesh and sampled down to 128×128 (and the
generator's own energy-balance check allows about 2.5% error), so a 128×128 finite-difference check cannot reproduce them exactly.
I have not proven this is the whole story. It means the physics term carries an inconsistency of roughly 8–12% at coarse scales, which limits
how much it can add, and it also explains why a stronger physics weight hurt.

Other reasons physics is not expected to help much in this setup: 500 labelled maps already contain the physics, and the heat source is
predicted by a second output head, so the residual only ties temperature to a predicted source. The survey (§4.2) says physics helps most when data is sparse or noisy;
our inputs are sparse, but the training labels are abundant. The scarcity test above did not show a gain even with 25 chips.

## 5. Figures

Heatmaps (ground truth, generated map, error map, four test chips, sensors marked) are in each run folder as `heatmaps.png`, for example
`results/fbpinn_fast_w16/N10_seed0/`, `results/xpinn_fast_seam/N10_seed0/`, `results/xpinn_fast_seam_pool16/N10_seed0/`, `results/fbpinn_fast/N8_seed0/`.
Generated with `python -m src.eval.heatmaps_fbpinn <best.pt> <out.png> 4`.

- FBPINN maps are smooth. In the 10-sensor full-grid FBPINN heatmap (`results/fbpinn/N10_seed0/`) the peaks are 12–14 °C low on three of the four chips,
  and the warm region is in roughly the right place.
- XPINN maps show visible blocky seams at the tile edges, because the interface loss does not enforce continuity exactly. Per-chip errors are 6–8% at 10 sensors.
- At 10 sensors the peak is still underestimated by about 7–9 °C on average for every model; the shortfall is larger at 8 and 5 sensors (peak error in `summary.md`).

## 6. Limitations

- Seeds: 10-sensor FBPINN and masked-physics XPINN have 3 seeds; 8-sensor FBPINN has 3; everything else (including the 16-window FBPINN, the physics-off runs
  and the scarcity runs) is a single seed. The DeepONet reference is the teammates' 5-seed mean, copied from their branch summary and not re-run here.
- Budgets differ by row (see `summary.md`): 5-sensor fast runs used 10,000 steps, the others 6,000, and the full-grid FBPINN and FNO runs used 3000 steps at batch 8.
- Hyper-parameters (window width, tile count, physics weight, sensor-encoder size) were only lightly explored. The physics weight 0.01 was not tuned per model.
- Physical constants (die thickness, heat-sink conductance, leakage coefficient) are flagged "ASSUMPTION — NOT SOURCED" in `PROTOCOL.md`.
- The existing `src/eval/gappy_pod.py` on `main` applies a basis fitted to normalised temperatures to raw readings, so its numbers are not meaningful; it was not used here.
- All runs were CPU-only and several shared the CPU, so timings are not clean.

## 7. Reproducing the main runs

```
# FBPINN (9 windows) and XPINN with masked physics, fast mode
python -m src.train.fbpinn --sensors 10 --seed 0 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --out results/fbpinn_fast
python -m src.train.xpinn  --sensors 10 --seed 0 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --seam-mask --out results/xpinn_fast_seam
# 16-window FBPINN
python -m src.train.fbpinn --sensors 10 --seed 0 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --windows 4 --window-width 0.25 --out results/fbpinn_fast_w16
# physics off
python -m src.train.xpinn  --sensors 10 --seed 0 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --w-phys 0 --out results/xpinn_fast_wphys0
# tables and figures
python -m src.eval.compare_models && python -m src.eval.scarcity_table && python -m src.eval.physics_consistency
```
