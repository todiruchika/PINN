# PINN Thermal Map — Evaluation Protocol

Each item is marked **DECIDED** or **PENDING**.

---

## Physics

Source: `generator/pde_solver.py` (verified 2026-10-13).

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
| Matrix dimensions | DECIDED | 16384 × 500 (flattened 128×128, 500 train samples). Layout refit on the 500-sample train split (r = 11 at 95% energy, `train_ids_sha256` 599325f6…); sensor files regenerated for all 700 samples with `python -m src.sensors.generate_sensors`. `generate_sensors` does not refit the layout — run `fit_podqr_layout` first if the train split changes. |
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
| `block_centre_reference` | DECIDED | Stored block-centre positions (block_centre_with_jitter) from the delivered dataset. |
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
| Split sizes | DECIDED | train=500, validation=100, test=100 (700 total). |
| Pixel-level leakage | DECIDED | Not split by pixel. |

### Sampling strategy

Describes what the generator code does. Sources: `generator/dataset_generator.py`, `generator/floorplan_generator.py`, and `generator/pde_solver.py`; labels from `generator/PARAMETERS.md`. Items the code does not establish are marked UNKNOWN.

| Item | Code behaviour | Source |
|------|----------------|--------|
| Die | 12 mm × 12 mm square (`DIE_LX = DIE_LY = 0.012` m). | floorplan_generator.py L36–37 |
| Ambient T_amb | Uniform in [25, 45] °C. Drawn in the parent process from the global NumPy RNG (seeded with `global_seed`), in order: all train, then val, then test. Changing one split's count changes T_amb for every later split. | dataset_generator.py L185; pde_solver.py L78–79 |
| Block count | Uniform integer in [2, 8]. Placement failures (200 tries per block) skip that block, so realised `block_count` can be lower. If zero blocks result, one retry with 2 blocks. Realised count is recorded per sample. | floorplan_generator.py L222, L229–231; dataset_generator.py L96–98 |
| Block types | Uniform choice among CPU, GPU, Cache, MC. | floorplan_generator.py L224, L228 |
| Block geometry | Sizes as fractions of die edge: CPU w,h ∈ [0.12, 0.25]; GPU w ∈ [0.20, 0.40], h ∈ [0.18, 0.35]; Cache w ∈ [0.18, 0.35], h ∈ [0.10, 0.28]; MC is a strip (one side ∈ [0.04, 0.09], other ∈ [0.25, 0.55]) on a randomly chosen die edge. Position uniform within the die. Blocks are rejection-sampled to be non-overlapping (5×10⁻⁵ m margin). | floorplan_generator.py L107–157 |
| Power levels | Per block, dynamic density p_dyn ~ U(range) in W/cm²: CPU [40, 150], GPU [30, 120], Cache [5, 20], MC [8, 30]. Leakage at 25 °C is p_dyn × fraction (CPU 0.05, GPU 0.05, Cache 0.04, MC 0.03). Leakage at T is p_leak0 · exp(β(T − 25)), β = 0.012 /K, re-evaluated in the Picard outer loop. | floorplan_generator.py L45–71, L233–236; pde_solver.py L150–164 |
| Power map | Each block contributes (p_dyn + p_leak) × 10⁴ W/m² over its footprint, weighted by cell-area overlap. | pde_solver.py L240–254 |
| Intra-block non-uniformity | `floorplan_generator` builds per-block Gaussian hot-spot weights (1–3 bumps), but `intra_weight` is not read by `pde_solver.py` or `dataset_generator.py`. The power map is therefore uniform within each block. | floorplan_generator.py L162–204; no consumers found |
| Floorplan seed | Per sample: `global_seed + floorplan_id × 100`. `floorplan_id` is a running index across splits (train 0…, then val, then test), so no floorplan is shared between splits. | dataset_generator.py L91–93, L187, L216–224 |
| Global seed | 42 (CLI `--seed`). Seeds the T_amb draws and floorplan/sensor RNGs. | dataset_generator.py L248, L64 |
| Thermal violation | Peak T > 125 °C is flagged in metadata; samples are not dropped. The 125 °C threshold is ASSUMPTION — NOT SOURCED (PARAMETERS.md §8). | dataset_generator.py L41–46, L148–150 |
| Default sample counts | Script default is "prototype" 500/100/100; "production" 50 000/5 000/5 000. Actual counts are set by CLI flags. | dataset_generator.py L240–261 |

**UNKNOWN**

- Whether the delivered 700 samples were produced by the current generator. `results/dataset_extension_report.md` records a historical regression check for samples 00001–00002, but there is no complete generation record for the delivered set.
- Provenance of the delivered `splits.csv` is unknown. The current generator writes a manifest for newly generated data; it does not regenerate the delivered split assignment.
- Block-count range: the `generate_single_sample` docstring says 1–10 blocks; the code samples 2–8. The code is used as the behavioural reference here.
- Power-density ranges, leakage fractions, T_amb bounds, and the 125 °C threshold are all labelled ASSUMPTION — NOT SOURCED in `generator/PARAMETERS.md`.
- Split naming: generated datasets use `train`, `validation`, and `test`, matching `splits.csv` and the data loader.

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
5. **700 samples** — Below the ≥ 2,000 target cited for the paper. Acquiring more data is a hard dependency for publication.
6. **Energy balance tolerance** — The validator now uses the generator convention (dA = (L/(N−1))²) and a 2.5% threshold. See `results/validation/energy_investigation.md` for the measured coarse-grid residual.

---

## Branch Plan

1. `main` — this protocol + data/sensor/eval harness. Merges when tests pass.
   Status (2026-10-13): contains `src/data`, `src/sensors`, `src/eval` (incl. `coverage.py`), `src/physics/heat.py` (stubs raising `NotImplementedError`), and `tests/`. `src/models/` and `src/train/` contain only `__init__.py`. The extended dataset is incomplete (28 of 400 train matrices), so no coverage comparison has been run against it yet.
2. Method branches (`Ruchika`, `smriti` exist on origin; siya's branch not yet pushed) start from `main` after this merges.
3. Each branch implements its model in `src/models/` and training in `src/train/`. No model code on `main`.
4. PRs back to `main` to integrate validated results.
