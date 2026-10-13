# PINN Thermal Map — Evaluation Protocol

Each item is marked **DECIDED** or **PENDING**.

---

## Physics

Source: `generator/package_for_claude/pde_solver.py` (verified 2026-10-13).

| Item | Status | Detail |
|------|--------|--------|
| Domain | DECIDED | Steady-state 2D heat conduction on a 12 mm × 12 mm die |
| Governing equation | DECIDED | −∇·(t_die · k(T) ∇T) + h_eff (T − T_amb) = q(x,y,T) — HotSpot compact 2D model (Huang et al., IEEE TVLSI 2006). |
| Lateral BCs | DECIDED | Neumann dT/dn = 0 on all four lateral edges (adiabatic). Source: `pde_solver.py` docstring; Therm-FM arXiv 2605.22663; HotSpot Huang et al. 2006. |
| Top-surface BC | DECIDED | Distributed vertical sink: h_eff(T − T_amb) added to every interior node. This h_eff term makes the system uniquely solvable without any Dirichlet constraint. |
| Thermal conductivity k(T) | DECIDED | k(T[K]) = 22 714 / (T[K] − 99) + 2, clipped to [20, 160] W/(m·K), valid for T ∈ [250, 1200] K. Source: Glassbrenner & Slack (1964). **ASSUMPTION**: coefficients (A=22714, offset=99, B=2) are a custom fit; quantitative validation against G&S 1964 data is pending. Source file: `pde_solver.py` constants `_K_A`, `_K_OFFSET`, `_K_B`. |
| Source term q | DECIDED | q = Σ_blocks (p_dyn + p_leak0 · exp(β(T − 25))) W/cm² × 10⁴ → W/m². Leakage is temperature-dependent and updated in the outer Picard loop. Stored as `thermal_NNNNN_Q.npy`. |
| Leakage coefficient β | DECIDED | β = 0.012 /K. **ASSUMPTION**: within literature range [0.010, 0.015] /K (He et al., DAC 2004; Liao et al., IEEE TCAD 2005) but not the exact value from either paper. Source file: `pde_solver.py` constant `LEAKAGE_BETA_DEFAULT`. |
| Die thickness t_die | DECIDED | t_die = 500 µm. **ASSUMPTION — NOT SOURCED**: VarSim (arXiv 2307.12119, Table II) uses 150 µm. Confirm before publication. Source file: `pde_solver.py` constant `DIE_THICKNESS_DEFAULT`. |
| Effective heat-sink conductance h_eff | DECIDED | h_eff = 6256.3 W/(m²·K) = 1 / ((R_die + R_spreader + R_heatsink) × A_chip). **ASSUMPTION**: R values are HotSpot defaults from `default.config`, not from the uploaded HotSpot 2006 paper text. Source file: `pde_solver.py` constant `H_EFF_DEFAULT`. |
| Solver | DECIDED | Picard iteration (inner loop: k(T) nonlinearity; outer loop: leakage update). Fine mesh 512×512 → downsampled to 128×128 by stride `[::4,::4]` (anti-inverse-crime, Wirgin arXiv math-ph/0401050). Solver tag: `FineMesh512_to_128`. |

---

## Nondimensionalisation

| Item | Status | Detail |
|------|--------|--------|
| θ = (T − T_amb) / (max T − T_amb) | DECIDED (2026-10-13) | Per sample. T_amb is per-sample ambient from metadata. max T is the sample's peak temperature. Normalises to [0, 1]. Used for POD-QR fitting only; all reported metrics remain in °C. |
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
| Normalisation | DECIDED (2026-10-13) | θ_i = (T_i − T_amb,i) / (max T_i − T_amb,i) per sample. T_amb from metadata column `T_amb_C`. |
| T_b rule | DECIDED (2026-10-13) | Use per-sample `T_amb_C` from metadata (consistent with the physics: T_amb is the known ambient temperature in the model). |
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

1. **BC resolved (2026-10-13)** — The BC is adiabatic Neumann (dT/dn = 0) on all lateral edges, plus a distributed h_eff sink to T_amb on every node. The apparent edge-temperature deviation from T_amb in the coarse field is a stride-downsampling artifact, not a Dirichlet BC violation.
2. **Governing equation resolved (2026-10-13)** — PDE confirmed: −∇·(t_die · k(T)∇T) + h_eff(T − T_amb) = q(x,y,T) with temperature-dependent k(T) and leakage. Source: `pde_solver.py`.
3. **k resolved (2026-10-13)** — k(T) = 22714/(T[K]−99) + 2, clipped to [20, 160] W/(m·K). Source: `pde_solver.py`. Note: k is NOT in the metadata CSV; it must be computed at runtime from each sample's T field.
4. **Three assumptions need external confirmation** — t_die = 500 µm, h_eff = 6256.3 W/(m²·K), β = 0.012 /K. All flagged as ASSUMPTION — NOT SOURCED in `generator/PARAMETERS.md`. Must confirm against HotSpot `default.config` and published data before submission.
5. **100 samples** — Far below ≥ 2,000 target cited for the paper. Acquiring more data is a hard dependency for publication.
6. **Energy balance tolerance** — Validator threshold (3%) needs updating to match the generator convention (dA = (L/(N−1))²) and the observed maximum (2.23%). See `results/validation/energy_investigation.md`.

---

## Branch Plan

1. `main` — this protocol + data/sensor/eval harness. Merges when tests pass.
2. Method branches (`Ruchika`, `smriti`, siya's branch) start from `main` after this merges.
3. Each branch implements its model in `src/models/` and training in `src/train/`. No model code on `main`.
4. PRs back to `main` to integrate validated results.
