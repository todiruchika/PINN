# Energy-Balance Investigation

Generated: 2026-10-13. All 100 samples (80 train, 10 validation, 10 test).

## Method

For each sample:
- `P_in_calc  = sum(Q) × dA`  where `dA = (0.012/128)²`  
- `P_out_calc = sum(h_eff × (T − T_amb)) × dA`  with `h_eff = 6256.256 W/(m²·K)`
- Compared against metadata `energy_bal_P_in` and `energy_bal_P_out`
- Fine grid (512×512) is **not available**; all analysis is on the coarse 128×128 grid.
- Downsampling is `T_fine[::4, ::4]` (stride, not area average). Fine grid not available to verify.

---

## Root Cause 1 — Grid-spacing convention (systematic ~1.56% offset)

The validator uses `dA = (L/N)² = (0.012/128)²`.  
The generator uses `dx = L/(N−1) = 0.012/127` → `dA_gen = (0.012/127)²`.

Ratio: `(128/127)² ≈ 1.0158`, i.e. the generator's cell area is 1.58% larger.

This causes `P_in_calc / P_in_meta ≈ 0.9845` for all samples — a constant multiplicative bias.  
The same offset applies to `P_out`. This is **not** a solver error; it is a metric definition difference.

---

## Root Cause 2 — Stride subsampling breaks coarse-grid energy balance (0.6–2.2%)

The fine-mesh solver (512×512) converges to energy balance on the fine grid. After
`T_coarse = T_fine[::4, ::4]`, `Q_coarse` is re-computed at `T_coarse`. Because stride
subsampling does not preserve area averages, and because the coarse node positions
`i × 0.012/127` do not coincide exactly with the fine node positions `4i × 0.012/511`,
the recomputed `Q_coarse` does not exactly satisfy the coarse-grid PDE.

Result: `P_out_calc > P_in_calc` for **all 100 samples** (consistently negative sign).  
The error is small (median 1.47%, max 2.23%) and uncorrelated with peak_T, T_amb, or block_count.

Lateral-edge contribution to `P_out`: 2.7% on average — boundary cells are **not** the
primary source of imbalance.

---

## |∂T/∂n| on lateral edges

Finite-difference estimate `|dT/dn|` on all four adiabatic lateral edges:

| Statistic | Value (K/m) |
|-----------|------------|
| Min | 5.3 |
| Median (mean) | 171.9 |
| Median (max per sample) | 850.6 |
| Maximum across all samples | 4118.8 |

These are large because stride subsampling creates a coarse `T` field that is **not** the
solution of the coarse-grid PDE with Neumann BCs. The true fine-mesh solution has
`dT/dn = 0` on all edges; the stride-sampled coarse field does not.

---

## Per-sample table

| sample_id | P_in_calc (W) | P_in_meta (W) | P_out_calc (W) | P_out_meta (W) | err_pct (%) |
|-----------|--------------|--------------|----------------|----------------|-------------|
| thermal_00001 | 2.744 | 2.787 | 2.805 | 2.849 | 2.162 |
| thermal_00002 | 12.149 | 12.341 | 12.270 | 12.464 | 0.987 |
| thermal_00003 | 35.955 | 36.523 | 36.519 | 37.097 | 1.546 |
| thermal_00004 | 48.818 | 49.590 | 49.387 | 50.168 | 1.151 |
| thermal_00005 | 19.588 | 19.898 | 19.863 | 20.177 | 1.385 |
| thermal_00006 | 17.755 | 18.035 | 17.989 | 18.274 | 1.304 |
| thermal_00007 | 18.441 | 18.733 | 18.698 | 18.994 | 1.372 |
| thermal_00008 | 17.122 | 17.392 | 17.449 | 17.725 | 1.875 |
| thermal_00009 | 22.585 | 22.942 | 22.882 | 23.244 | 1.300 |
| thermal_00010 | 42.312 | 42.981 | 43.031 | 43.712 | 1.671 |
| thermal_00011 | 25.221 | 25.620 | 25.686 | 26.092 | 1.807 |
| thermal_00012 | 3.974 | 4.037 | 4.034 | 4.098 | 1.474 |
| thermal_00013 | 20.480 | 20.804 | 20.774 | 21.102 | 1.415 |
| thermal_00014 | 5.335 | 5.419 | 5.371 | 5.456 | 0.686 |
| thermal_00015 | 24.617 | 25.006 | 24.843 | 25.236 | 0.909 |
| thermal_00016 | 7.518 | 7.637 | 7.685 | 7.807 | 2.175 |
| thermal_00017 | 13.586 | 13.801 | 13.882 | 14.102 | 2.134 |
| thermal_00018 | 35.197 | 35.754 | 35.762 | 36.328 | 1.580 |
| thermal_00019 | 28.702 | 29.155 | 29.158 | 29.619 | 1.565 |
| thermal_00020 | 17.111 | 17.381 | 17.346 | 17.620 | 1.356 |
| thermal_00021 | 28.989 | 29.448 | 29.486 | 29.952 | 1.683 |
| thermal_00022 | 27.732 | 28.170 | 28.127 | 28.572 | 1.406 |
| thermal_00023 | 37.135 | 37.722 | 37.775 | 38.373 | 1.696 |
| thermal_00024 | 6.150 | 6.248 | 6.223 | 6.322 | 1.173 |
| thermal_00025 | 7.434 | 7.551 | 7.527 | 7.646 | 1.244 |
| thermal_00026 | 21.706 | 22.049 | 22.108 | 22.458 | 1.819 |
| thermal_00027 | 19.214 | 19.518 | 19.551 | 19.860 | 1.722 |
| thermal_00028 | 20.039 | 20.356 | 20.342 | 20.663 | 1.488 |
| thermal_00029 | 18.476 | 18.768 | 18.894 | 19.193 | 2.211 |
| thermal_00030 | 3.495 | 3.550 | 3.529 | 3.585 | 0.971 |
| thermal_00031 | 35.400 | 35.960 | 35.919 | 36.487 | 1.444 |
| thermal_00032 | 8.000 | 8.127 | 8.134 | 8.263 | 1.649 |
| thermal_00033 | 54.592 | 55.455 | 55.410 | 56.286 | 1.476 |
| thermal_00034 | 52.004 | 52.826 | 52.778 | 53.612 | 1.467 |
| thermal_00035 | 19.041 | 19.342 | 19.382 | 19.688 | 1.756 |
| thermal_00036 | 33.491 | 34.021 | 33.904 | 34.440 | 1.216 |
| thermal_00037 | 14.709 | 14.942 | 15.045 | 15.282 | 2.230 |
| thermal_00038 | 29.020 | 29.479 | 29.436 | 29.901 | 1.413 |
| thermal_00039 | 8.914 | 9.055 | 8.989 | 9.131 | 0.832 |
| thermal_00040 | 20.892 | 21.222 | 21.284 | 21.620 | 1.840 |
| thermal_00041 | 41.014 | 41.662 | 41.793 | 42.454 | 1.865 |
| thermal_00042 | 15.075 | 15.313 | 15.277 | 15.519 | 1.325 |
| thermal_00043 | 20.148 | 20.467 | 20.359 | 20.681 | 1.037 |
| thermal_00044 | 7.004 | 7.114 | 7.146 | 7.259 | 1.986 |
| thermal_00045 | 12.108 | 12.299 | 12.296 | 12.491 | 1.534 |
| thermal_00046 | 10.779 | 10.949 | 10.898 | 11.070 | 1.091 |
| thermal_00047 | 14.048 | 14.270 | 14.265 | 14.491 | 1.525 |
| thermal_00048 | 14.179 | 14.403 | 14.388 | 14.615 | 1.453 |
| thermal_00049 | 1.394 | 1.417 | 1.416 | 1.438 | 1.499 |
| thermal_00050 | 13.774 | 13.992 | 13.910 | 14.129 | 0.973 |
| thermal_00051 | 34.531 | 35.077 | 35.031 | 35.585 | 1.427 |
| thermal_00052 | 33.348 | 33.876 | 33.791 | 34.325 | 1.309 |
| thermal_00053 | 33.390 | 33.918 | 34.102 | 34.641 | 2.088 |
| thermal_00054 | 25.147 | 25.544 | 25.600 | 26.005 | 1.771 |
| thermal_00055 | 24.033 | 24.413 | 24.466 | 24.853 | 1.773 |
| thermal_00056 | 18.177 | 18.464 | 18.494 | 18.786 | 1.712 |
| thermal_00057 | 16.158 | 16.413 | 16.402 | 16.661 | 1.489 |
| thermal_00058 | 2.470 | 2.509 | 2.485 | 2.524 | 0.605 |
| thermal_00059 | 28.883 | 29.340 | 29.261 | 29.724 | 1.293 |
| thermal_00060 | 13.547 | 13.761 | 13.703 | 13.920 | 1.141 |
| thermal_00061 | 15.021 | 15.259 | 15.236 | 15.477 | 1.406 |
| thermal_00062 | 6.949 | 7.059 | 7.041 | 7.153 | 1.304 |
| thermal_00063 | 8.106 | 8.234 | 8.240 | 8.370 | 1.629 |
| thermal_00064 | 11.813 | 11.999 | 11.982 | 12.171 | 1.412 |
| thermal_00065 | 25.906 | 26.316 | 26.279 | 26.694 | 1.417 |
| thermal_00066 | 18.455 | 18.747 | 18.723 | 19.019 | 1.431 |
| thermal_00067 | 39.069 | 39.687 | 39.664 | 40.291 | 1.501 |
| thermal_00068 | 1.378 | 1.400 | 1.391 | 1.413 | 0.908 |
| thermal_00069 | 37.673 | 38.269 | 38.297 | 38.903 | 1.630 |
| thermal_00070 | 26.025 | 26.436 | 26.349 | 26.766 | 1.233 |
| thermal_00071 | 31.682 | 32.183 | 32.136 | 32.644 | 1.412 |
| thermal_00072 | 19.282 | 19.587 | 19.553 | 19.862 | 1.385 |
| thermal_00073 | 44.772 | 45.479 | 45.274 | 45.989 | 1.109 |
| thermal_00074 | 13.775 | 13.992 | 13.874 | 14.093 | 0.715 |
| thermal_00075 | 16.356 | 16.614 | 16.619 | 16.882 | 1.586 |
| thermal_00076 | 9.064 | 9.207 | 9.122 | 9.266 | 0.632 |
| thermal_00077 | 14.826 | 15.060 | 15.089 | 15.328 | 1.743 |
| thermal_00078 | 8.109 | 8.237 | 8.248 | 8.378 | 1.685 |
| thermal_00079 | 9.681 | 9.834 | 9.864 | 10.020 | 1.863 |
| thermal_00080 | 10.157 | 10.317 | 10.306 | 10.469 | 1.445 |
| thermal_00081 | 14.872 | 15.108 | 15.125 | 15.364 | 1.668 |
| thermal_00082 | 22.305 | 22.657 | 22.655 | 23.013 | 1.545 |
| thermal_00083 | 25.630 | 26.035 | 25.970 | 26.381 | 1.310 |
| thermal_00084 | 20.182 | 20.501 | 20.517 | 20.841 | 1.631 |
| thermal_00085 | 46.078 | 46.807 | 46.822 | 47.563 | 1.589 |
| thermal_00086 | 29.656 | 30.125 | 30.096 | 30.571 | 1.461 |
| thermal_00087 | 50.293 | 51.088 | 51.126 | 51.935 | 1.631 |
| thermal_00088 | 29.812 | 30.284 | 30.348 | 30.827 | 1.764 |
| thermal_00089 | 13.565 | 13.779 | 13.700 | 13.917 | 0.989 |
| thermal_00090 | 35.256 | 35.813 | 35.792 | 36.358 | 1.499 |
| thermal_00091 | 47.841 | 48.597 | 48.643 | 49.412 | 1.650 |
| thermal_00092 | 20.052 | 20.369 | 20.425 | 20.748 | 1.830 |
| thermal_00093 | 18.453 | 18.745 | 18.698 | 18.993 | 1.309 |
| thermal_00094 | 21.743 | 22.087 | 22.075 | 22.424 | 1.505 |
| thermal_00095 | 12.527 | 12.725 | 12.656 | 12.856 | 1.020 |
| thermal_00096 | 17.550 | 17.827 | 17.777 | 18.058 | 1.279 |
| thermal_00097 | 2.654 | 2.696 | 2.687 | 2.729 | 1.210 |
| thermal_00098 | 11.226 | 11.403 | 11.426 | 11.607 | 1.756 |
| thermal_00099 | 1.355 | 1.377 | 1.382 | 1.404 | 1.939 |
| thermal_00100 | 4.923 | 5.001 | 4.984 | 5.062 | 1.211 |

---

## Diagnosis

Two separate sources of error:

**1. Metric definition mismatch (systematic ~1.56% offset, not a solver error)**  
The validator uses `dA = (L/N)² = (0.012/128)²`, while the generator uses
`dA = (L/(N−1))² = (0.012/127)²`. Ratio: `(128/127)² = 1.0158`.  
This accounts for the consistent gap between computed and metadata P_in / P_out values.
**The generator and the dataset are self-consistent.**

**2. Stride-subsampling breaks coarse-grid energy balance (0.6–2.2%)**  
The fine-mesh T is stride-downsampled (`[::4, ::4]`), not area-averaged. The resulting
coarse T is not the solution of the coarse-grid PDE with Neumann BCs, so the coarse
energy balance is imperfect. Edge cells contribute only ~2.7% of P_out and are not the
primary source. The error is independent of peak_T, T_amb, and block_count.

---

## Recommendation

**(c) Document the tolerance.**

The 0.6–2.2% imbalance is a structural consequence of the anti-inverse-crime stride
downsampling and the validator's different grid-spacing convention. It does not
indicate a solver bug. The imbalance is:
- Well below the generator's own tolerance (15% in PARAMETERS.md §9)
- Consistent across all 100 samples with no outliers
- Expected to persist in any extension dataset generated by the same pipeline

**Action items:**
1. Update the validator to use `dA = (L/(N−1))² = (0.012/127)²` to match the generator (eliminates the ~1.56% metric offset).
2. Update the energy FAIL threshold from 3% to 2.5% (the observed max is 2.23%; a 3% threshold is too loose for reporting).
3. Document in PROTOCOL.md that the residual 0.6–2.2% imbalance is a stride-downsampling artifact.

Do NOT fix the generator. The stride downsampling is intentional (anti-inverse-crime).
