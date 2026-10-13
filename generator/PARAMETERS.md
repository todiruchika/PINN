# PARAMETERS.md — PINN Thermal Dataset Pipeline

All numeric inputs used by the pipeline. For each parameter: value, unit, and
either an explicit source or **ASSUMPTION — NOT SOURCED** if no traceable citation
is available in the reference papers or standard textbooks provided.

No citations were invented. Sources marked with a paper, section, table, or page
are traceable to the uploaded reference material listed below. Parameters that
the authors could not match to a specific passage are labelled accordingly.

**Reference papers available for this pipeline:**
- Huang et al., *IEEE Trans. VLSI Syst.*, 14(9), 2006 (HotSpot). ("HotSpot 2006")
- Li et al., *MICRO 2009*, McPAT. ("McPAT 2009")
- Liao, He & Lepak, *IEEE Trans. CAD*, 24(9), 2005. ("TCAD 2005")
- He et al., *DAC 2004*. ("DAC 2004")
- VarSim, arXiv 2307.12119. ("VarSim")
- Therm-FM, arXiv 2605.22663. ("Therm-FM")
- Glassbrenner & Slack, *Phys. Rev.* 134, A1058, 1964. ("G&S 1964")
- Wirgin, arXiv math-ph/0401050. ("Wirgin")
- Intel Core Duo thermal sensors, arXiv 0709.1861. ("Core Duo 2007")
- Intel Agilex 7 Thermal Design Guide AN 944. ("Agilex AN 944")

---

## 1. Die Geometry

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| Die lateral dimension L_x | 0.012 | m | **ASSUMPTION — NOT SOURCED.** 12 mm × 12 mm is a representative mid-range die footprint used for dataset generation. The HotSpot EV6 floorplan (Huang et al. 2006) uses ~16 mm × 16 mm. No specific 12 mm die dimension is cited in any uploaded paper. |
| Die lateral dimension L_y | 0.012 | m | Same as L_x above. **ASSUMPTION — NOT SOURCED.** |
| Die thickness t_die | 500 × 10⁻⁶ | m | **ASSUMPTION — NOT SOURCED.** VarSim (arXiv 2307.12119, Table II) uses 0.15 mm (150 µm). The 0.5 mm value used here is broader and was not read from any uploaded paper. Confirm before publication. |

---

## 2. Package Thermal Resistance Model

The effective vertical heat-sink conductance h_eff is derived from the HotSpot
compact resistance network. The formula is:

```
h_eff = 1 / (R_total × A_chip)
R_total = R_die + R_spreader + R_heatsink
A_chip  = L_x × L_y = 0.012 × 0.012 = 1.44 × 10⁻⁴ m²
```

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| R_die (TIM contact resistance) | 0.01 | °C/W | **ASSUMPTION — NOT SOURCED.** Listed as a placeholder derived from the HotSpot default.config file comment in the codebase. The uploaded HotSpot 2006 paper (Huang et al.) does not reproduce the exact numeric default.config values in the uploaded text. The HotSpot tool distribution's `default.config` file is the authoritative source; it was not uploaded. |
| R_spreader (heat-spreader resistance) | 0.100 | °C/W | **ASSUMPTION — NOT SOURCED.** Same reason as R_die above. |
| R_heatsink (heatsink-to-air resistance) | 1.000 | °C/W | **ASSUMPTION — NOT SOURCED.** Same reason. Huang et al. (HotSpot 2006) describes the resistance network topology in §II but the exact numeric defaults appear in the tool's `default.config`, not in the uploaded paper text. |
| h_eff (derived) | 6 256.3 | W/(m²·K) | **ASSUMPTION — NOT SOURCED** (derived from the three ASSUMPTION values above). Formula: 1 / ((0.01 + 0.10 + 1.00) × 1.44 × 10⁻⁴) = 6 256.3 W/(m²·K). If the source R values change, h_eff changes proportionally. |

---

## 3. Temperature-Dependent Thermal Conductivity k(T)

Model: k(T) = A / (T[K] − T_offset) + B

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| Conductivity coefficient A | 22 714 | W·K/m | **ASSUMPTION — NOT SOURCED.** The k(T) Umklapp-scattering functional form follows G&S 1964 (k ~ 1/T for crystalline Si above ~100 K). The fit coefficients A, T_offset, B are a custom fit used by this pipeline whose residuals have not been published. The fit form is consistent with G&S 1964 and Touloukian et al. (1970) but the exact coefficients were not read from those papers. |
| Conductivity offset T_offset | 99 | K | Same as A above. **ASSUMPTION — NOT SOURCED.** |
| Conductivity floor B | 2.0 | W/(m·K) | Same as A above. **ASSUMPTION — NOT SOURCED.** |
| Valid T range (clip) | 250–1200 | K | Conservative physical range for crystalline Si; clipping behaviour only. G&S 1964 data range is ~2–1400 K. |
| k clamp lower | 20.0 | W/(m·K) | Numerical guard only. |
| k clamp upper | 160.0 | W/(m·K) | Numerical guard only (Si at ~200 K). |
| k at 300 K (26.85 °C) | 115.0 | W/(m·K) | Formula-derived (not independently verified against G&S 1964 Table I). |
| k at 300.15 K (27.0 °C) | 114.9 | W/(m·K) | Formula-derived (22714 / (300.15 − 99) + 2.0 = 114.92 W/(m·K)). |

---

## 4. Leakage Power Model

Model: P_leak(T) = P_leak0 × exp(β × (T − T₀))

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| Leakage exponential coefficient β | 0.012 | 1/K | **ASSUMPTION — NOT SOURCED** (value selection). He et al. (DAC 2004) describe exponential subthreshold leakage growth with temperature and report that leakage roughly triples from 25 °C to 100 °C, yielding β ≈ ln(3)/75 ≈ 0.0146 /K. Liao et al. (TCAD 2005) use a similar model. The value 0.012 /K is within the quoted range [0.010, 0.015] /K but is not the exact value from either paper. |
| Leakage reference temperature T₀ | 25.0 | °C | Standard JEDEC ambient reference temperature. No specific uploaded paper was consulted; this is a well-established convention. **ASSUMPTION — NOT SOURCED** (no uploaded paper cited). |
| Leakage fraction at T₀ — CPU | 5.0 | % of P_dyn | **ASSUMPTION — NOT SOURCED.** McPAT (Li et al., MICRO 2009) reports technology-node-dependent leakage fractions but the exact per-block percentage at 25 °C for the chosen process node was not read from the uploaded McPAT paper. |
| Leakage fraction at T₀ — GPU | 5.0 | % of P_dyn | Same as CPU. **ASSUMPTION — NOT SOURCED.** |
| Leakage fraction at T₀ — Cache | 4.0 | % of P_dyn | Same. **ASSUMPTION — NOT SOURCED.** |
| Leakage fraction at T₀ — MC | 3.0 | % of P_dyn | Same. **ASSUMPTION — NOT SOURCED.** |
| Exponent overflow guard (clamp) | ±20 | — | Numerical guard; no physical significance. |

---

## 5. Power Density Ranges

All values in W/cm². The ranges are representative engineering estimates intended to produce
realistic temperature maps. They are **not read from any uploaded paper**.

| Block type | Min | Max | Unit | Source or Label |
|---|---|---|---|---|
| CPU | 40.0 | 150.0 | W/cm² | **ASSUMPTION — NOT SOURCED.** McPAT (Li et al., MICRO 2009) provides power breakdowns for CPUs but per-cm² densities depend on die size and process node, which were not extracted from the uploaded paper. |
| GPU | 30.0 | 120.0 | W/cm² | **ASSUMPTION — NOT SOURCED.** Same reason. |
| Cache (SRAM) | 5.0 | 20.0 | W/cm² | **ASSUMPTION — NOT SOURCED.** SRAM arrays have lower power density; 5–20 W/cm² is a plausible estimate but was not read from any uploaded paper. |
| Memory Controller (MC) | 8.0 | 30.0 | W/cm² | **ASSUMPTION — NOT SOURCED.** Same. |

---

## 6. Sensor Model

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| Sensor noise sigma (default) | 1.0 | °C | **ASSUMPTION — NOT SOURCED** (value chosen between two conflicting specs). Intel Core Duo on-die sensors (arXiv 0709.1861, §3) report ±0.5 °C A/D resolution. Intel Agilex 7 AN 944 reports ±5 °C digital sensor accuracy. The 1.0 °C default is a designer choice between these two bounds; it was not specified in any uploaded paper. |
| Sigma (analogue reference) | 0.5 | °C | Source: arXiv 0709.1861 (Intel Core Duo on-die thermal sensor characterisation, §3 and Table I). Describes A/D quantization and analogue noise floor. |
| Sigma (digital reference) | 5.0 | °C | Source: Intel Agilex 7 Thermal Design Guide AN 944. Describes digital thermal sensor (DTS) accuracy as approximately ±5 °C. |
| ADC quantization step | 1.0 | °C | Source: arXiv 0709.1861, §3 (Intel Core Duo on-die thermal sensors): "The digital output has 1 °C resolution." |
| Calibration offset range | ±2.0 | °C | **ASSUMPTION — NOT SOURCED.** Sensor-to-sensor calibration spread of ±2 °C per sensor is a modelling choice not read from any uploaded paper. |
| Sensor count tiers | 5, 8, 10 | sensors | **ASSUMPTION — NOT SOURCED.** No uploaded paper specifies these three tiers. |
| Sensor boundary margin | 3 | grid cells | **ASSUMPTION — NOT SOURCED.** Implementation guard to avoid boundary artefacts. |
| Sensor placement strategy | Block centres ± 10% of half-width | — | Source: arXiv 0709.1861, §3.1: "sensors are placed at design time near expected hot spots but do not always capture the hottest point." Implemented as block-centre-based placement with jitter. |

---

## 7. Ambient Temperature

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| T_amb minimum | 25.0 | °C | **ASSUMPTION — NOT SOURCED.** Standard JEDEC JESD51 ambient reference temperature is 25 °C. No specific page from an uploaded paper was consulted. |
| T_amb maximum | 45.0 | °C | **ASSUMPTION — NOT SOURCED.** The upper bound of 45 °C is a design choice for dataset diversity; no uploaded paper specifies this range. |

---

## 8. Thermal Violation and Junction Temperature

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| T_violation threshold (T_VIOLATION_C) | 125.0 | °C | **ASSUMPTION — NOT SOURCED.** 125 °C is a widely quoted JEDEC maximum junction temperature, but the specific JEDEC standard was not uploaded. No uploaded paper cites this value. |
| Silicon melting point | 1412.0 | °C | Well-known physical constant. No specific paper needed. |

---

## 9. Numerical / Solver Parameters

| Parameter | Value | Unit | Source or Label |
|---|---|---|---|
| Grid size (coarse solver) | 128 × 128 | nodes | **ASSUMPTION — NOT SOURCED.** Chosen to balance compute cost and resolution. |
| Grid size (fine reference solver) | 512 × 512 | nodes | **ASSUMPTION — NOT SOURCED.** FINE_FACTOR = 4 × coarse. |
| Picard inner loop tolerance | 0.1 | K | **ASSUMPTION — NOT SOURCED.** |
| Picard max inner iterations | 20 | — | **ASSUMPTION — NOT SOURCED.** |
| Outer (thermo) loop tolerance | 0.1 | K | **ASSUMPTION — NOT SOURCED.** |
| Outer (thermo) max iterations | 15 | — | **ASSUMPTION — NOT SOURCED.** |
| Energy balance tolerance (validator) | 15.0 | % | **ASSUMPTION — NOT SOURCED.** Chosen to be consistent with compact model error levels; no specific paper quantifies this. |
| Grid-convergence tolerance (validator) | 2.0 | % | **ASSUMPTION — NOT SOURCED.** |
| Sensor noise bound (4σ rule) | 4.0 | σ | **ASSUMPTION — NOT SOURCED.** 4σ gives ~99.994% coverage for Gaussian noise. |
| Anti-inverse-crime separation | FineMesh (512) vs PINN (128) | — | Source: Wirgin (arXiv math-ph/0401050): "The inverse crime consists of using the same model for data generation and inversion." |

---

## 10. Summary: Parameters Labelled ASSUMPTION — NOT SOURCED

The following parameters have no traceable source in the uploaded papers and must
be confirmed or replaced before submission:

1. `DIE_LX_DEFAULT`, `DIE_LY_DEFAULT` — die dimensions (12 mm × 12 mm)
2. `DIE_THICKNESS_DEFAULT` — die thickness (500 µm; VarSim Table II uses 150 µm)
3. `R_DIE_DEFAULT`, `R_SPREADER_DEFAULT`, `R_HEATSINK_DEFAULT` — HotSpot resistance defaults (need HotSpot `default.config`)
4. `H_EFF_DEFAULT` (derived from the ASSUMPTION resistances above)
5. `_K_A`, `_K_OFFSET`, `_K_B` — conductivity fit coefficients
6. `LEAKAGE_BETA_DEFAULT` — leakage exponent 0.012 /K (within range, but not exact value from any paper)
7. `LEAKAGE_T0` — leakage reference temperature 25 °C
8. `LEAKAGE_FRACTION` — per-block leakage fractions (5%, 5%, 4%, 3%)
9. `POWER_DENSITY_RANGES` — per-block power density ranges (all four block types)
10. `SIGMA_DEFAULT_C` — sensor noise 1.0 °C (between Core Duo 0.5 °C and Agilex 5.0 °C)
11. `CALIB_OFFSET_RANGE_C` — calibration offset ±2.0 °C
12. `T_AMB_MIN_C`, `T_AMB_MAX_C` — ambient temperature sampling range
13. `T_VIOLATION_C` — thermal violation threshold 125 °C (JEDEC; uploaded paper not available)
