# Dataset Validation Report

| Check | Status | Message |
|-------|--------|---------|
| Layout | PASS | data/raw/dataset/train/matrices exists |
| Layout | PASS | data/raw/dataset/train/images exists |
| Layout | PASS | data/raw/dataset/train/sensors exists |
| Layout | PASS | data/raw/dataset/validation/matrices exists |
| Layout | PASS | data/raw/dataset/validation/images exists |
| Layout | PASS | data/raw/dataset/validation/sensors exists |
| Layout | PASS | data/raw/dataset/test/matrices exists |
| Layout | PASS | data/raw/dataset/test/images exists |
| Layout | PASS | data/raw/dataset/test/sensors exists |
| Layout.IDs | PASS | 700 IDs consistent across metadata and splits.csv |
| Layout.train.matrix_ids | PASS | 500 IDs match |
| Layout.validation.matrix_ids | PASS | 100 IDs match |
| Layout.test.matrix_ids | PASS | 100 IDs match |
| Splits.counts | PASS | train=500, validation=100, test=100 |
| Splits.overlap | PASS | No overlap across splits |
| Splits.train.meta_agree | PASS | 500 IDs agree |
| Splits.validation.meta_agree | PASS | 100 IDs agree |
| Splits.test.meta_agree | PASS | 100 IDs agree |
| Grids | PASS | All grids (128,128), finite, Q>=0 |
| Temperature | PASS | peak_T: min=30.3 median=83.9 max=124.6 degC; 0 samples > 125 degC |
| Boundary.ZeroFlux | WARN | Lateral-edge |dT/dn|: mean=170.1 K/m, max=3498.0 K/m. Non-zero values are expected (stride-downsampling artifact; BC is adiabatic Neumann). |
| Energy | PASS | max |ratio-1| = 0.0001, median = 0.0000 |
| Energy.err_pct | WARN | energy_bal_err_pct max = 2.71% > 2.5% |
| Sensors | PASS | All N in {5,8,10} present, positions in [0,0.012] |
| Sensors.placement | WARN | Stored reference sensors use block_centre_with_jitter (not podqr) |
| Metadata.conductivity | WARN | k not in metadata — expected. Compute at runtime: k(T[K]) = 22714/(T[K]-99)+2 (source: pde_solver.py). See PROTOCOL.md. |
| Metadata.BC | WARN | boundary_condition not in metadata — expected. BC is adiabatic Neumann on all lateral edges + h_eff distributed sink. Source: pde_solver.py. |
