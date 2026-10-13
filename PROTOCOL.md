# PINN Thermal Map — Evaluation Protocol

Each item is marked **DECIDED** or **PENDING**.

---

## Physics

| Item | Status | Detail |
|------|--------|--------|
| Domain | DECIDED | Steady-state 2D heat conduction on a 12 mm × 12 mm die |
| Source term | DECIDED | q in W/m², stored as `thermal_NNNNN_Q.npy` |
| Governing equation | PENDING | Linear conduction (-k∇²T = Q) vs leakage model (-k∇²T = Q_dyn + β·(T−T_amb)·Q_leak0) with β = 0.012. Newton iterations = 1 for all 100 samples; whether this distinguishes the two forms is unconfirmed. |
| Boundary condition | PENDING | **BLOCKER** — edge temperatures deviate by up to 92 °C from `T_amb_C`. The BC is not Dirichlet at T_amb. Unknown. |
| Thermal conductivity k | PENDING | **BLOCKER** — not present in metadata. |

---

## Nondimensionalisation

| Item | Status | Detail |
|------|--------|--------|
| θ = (T − T_ref)/ΔT_ref | PENDING | Cannot define until BC is known. |
| Error reporting | DECIDED | Report all errors in °C (absolute), plus relative L2. |

---

## Ground Truth

| Item | Status | Detail |
|------|--------|--------|
| Source | DECIDED | FEM on FineMesh512, resampled to 128×128. |
| Solver tag | DECIDED | `FineMesh512_to_128` (recorded per sample in metadata). |

---

## Evaluation Grid

| Item | Status | Detail |
|------|--------|--------|
| Grid size | DECIDED | 128×128. |
| Masking | DECIDED | Metrics computed on **non-sensor nodes only**. All sensor nodes masked for each N ∈ {5, 8, 10}. |

---

## Sensor Inputs

| Item | Status | Detail |
|------|--------|--------|
| Primary input | DECIDED | `temperatures_noisy` only. |
| Fidelity check | DECIDED | `temperatures_true` used only for sensor-level accuracy verification. |

---

## Sensor Placement — POD-QR

Reference: Manohar, Brunton, Kutz & Brunton, IEEE Control Systems Magazine 2018; Li, Liu & Wang, Science China Technological Sciences 2021.

| Item | Status | Detail |
|------|--------|--------|
| Split used for fitting | DECIDED | Training split only. Validation and test splits never opened during layout computation. |
| Normalisation | DECIDED | θ_i = (T_i − T_b,i) / (max T_i − T_b,i) per sample. |
| T_b rule | PENDING | **Default**: mean of four edge temperatures of each sample. Revisit when BC is known. |
| Matrix dimensions | DECIDED | 16384 × 80 (flattened 128×128, 80 train samples). |
| SVD centring | DECIDED | Subtract training mean before SVD. |
| Mode count r | DECIDED | Keep r modes where cumulative energy ≥ 95%. Sensitivity reported at 90% and 99%. |
| QR pivoting | DECIDED | QR with column pivoting on Ψ_r^T. First 10 pivots → sensors. Nested by construction. |
| Edge exclusion | DECIDED | Exclude pixels within 3 cells of any edge. |
| Sensor counts | DECIDED | N ∈ {5, 8, 10}. Nesting: {5} ⊂ {8} ⊂ {10}. |

---

## Comparison Arms

| Arm | Status | Detail |
|-----|--------|--------|
| `podqr` | DECIDED | Primary strategy. Deterministic. |
| `random_uniform` | DECIDED | Seeded uniform random within margin. |
| `block_centre_legacy` | DECIDED | Legacy positions from raw dataset (block_centre_with_jitter). |
| `kmeans_floorplan` | DECIDED (layout only) | K-means seeded from floorplan centroids. Layout only — never a model input. **Stub** pending floorplan metadata parsing convention. |

---

## Noise Model

| Item | Status | Detail |
|------|--------|--------|
| Type | DECIDED | Gaussian σ = 1.0 °C, quantised to 1.0 °C steps. |
| Calibration offsets | DECIDED | N(0, 0.5 °C) per sensor index, global seed 42. |
| Per-sample seed | DECIDED | Derived from sample ID numeric part and global seed. |

---

## Splits

| Item | Status | Detail |
|------|--------|--------|
| Source | DECIDED | `splits.csv` as given. Do not regenerate. |
| Split sizes | DECIDED | train=80, validation=10, test=10. |
| Pixel-level leakage | DECIDED | Not split by pixel. |

---

## Metrics (non-sensor nodes)

| Metric | Status | Detail |
|--------|--------|--------|
| Relative L2 | DECIDED | ‖pred−true‖₂ / ‖true‖₂ |
| MAE | DECIDED | °C |
| RMSE | DECIDED | °C |
| MSE | DECIDED | °C² |
| Peak-T error | DECIDED | °C |
| Hotspot localisation | DECIDED | grid cells (Euclidean) |
| Boundary max error | PENDING | Needs BC |
| PDE residual RMS | PENDING | Needs BC and governing equation |
| Inference time | DECIDED | seconds |
| Parameter count | DECIDED | per model |
| Training wall-clock | DECIDED | hours |
| Hardware | DECIDED | CPU/GPU model reported |

---

## Non-Neural Baseline

| Item | Status | Detail |
|------|--------|--------|
| Method | DECIDED | Gappy POD reconstruction using saved basis from POD-QR fitting. |
| Implementation | DECIDED | `src/eval/gappy_pod.py` |

---

## Seeds and Variance

| Item | Status | Detail |
|------|--------|--------|
| Seeds | DECIDED | 5 seeds (0–4) per model and configuration. |
| Reporting | DECIDED | Mean ± std over seeds. |
| Layout seed variance | DECIDED | None — POD-QR layout is deterministic. |

---

## Reporting Checklist (Table 5 from survey)

- [ ] Architecture and parameter count
- [ ] Optimizer and schedule
- [ ] Collocation/data point counts
- [ ] Error definition (masked grid, non-sensor nodes)
- [ ] Training budget (wall-clock, hardware)
- [ ] Independent seeds (5) and variance
- [ ] Hyperparameter-tuning parity across arms

---

## Blockers

1. **BC unknown** — Edge temperatures deviate from T_amb by up to 92 °C. Must resolve before: PDE residual, hard-BC ansatz, nondimensionalisation.
2. **Governing equation** — Linear vs leakage (β = 0.012). Must confirm before implementing physics loss.
3. **k not in metadata** — Cannot compute PDE residual.
4. **100 samples** — Far below ≥ 2,000 target cited for the paper. Acquiring more data is a hard dependency for publication.

---

## Branch Plan

1. `main` — this protocol + data/sensor/eval harness. Merges when tests pass.
2. Method branches (`Ruchika`, `smriti`, siya's branch) start from `main` after this merges.
3. Each branch implements its model in `src/models/` and training in `src/train/`. No model code on `main`.
4. PRs back to `main` to integrate validated results.
