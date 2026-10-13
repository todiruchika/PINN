# Dataset Validation Report

| Check | Status | Message |
|-------|--------|---------|
| Layout | PASS | data\raw\dataset\train\matrices exists |
| Layout | PASS | data\raw\dataset\train\images exists |
| Layout | PASS | data\raw\dataset\train\sensors exists |
| Layout | PASS | data\raw\dataset\validation\matrices exists |
| Layout | PASS | data\raw\dataset\validation\images exists |
| Layout | PASS | data\raw\dataset\validation\sensors exists |
| Layout | PASS | data\raw\dataset\test\matrices exists |
| Layout | PASS | data\raw\dataset\test\images exists |
| Layout | PASS | data\raw\dataset\test\sensors exists |
| Layout.IDs | PASS | 100 IDs consistent across metadata and splits.csv |
| Layout.train.matrix_ids | PASS | 80 IDs match |
| Layout.validation.matrix_ids | PASS | 10 IDs match |
| Layout.test.matrix_ids | PASS | 10 IDs match |
| Splits.counts | PASS | train=80, validation=10, test=10 |
| Splits.overlap | PASS | No overlap across splits |
| Splits.train.meta_agree | PASS | 80 IDs agree |
| Splits.validation.meta_agree | PASS | 10 IDs agree |
| Splits.test.meta_agree | PASS | 10 IDs agree |
| Grids | PASS | All grids (128,128), finite, Q>=0 |
| Temperature | PASS | peak_T: min=31.2 median=89.1 max=164.2 degC; 11 samples > 125 degC |
| Boundary | FAIL | BLOCKER: max edge deviation from T_amb = 121.09 degC (mean 47.33 degC). BC is unknown. |
| Energy | PASS | max |ratio-1| = 0.0156, median = 0.0156 |
| Energy.err_pct | PASS | energy_bal_err_pct max = 2.23% |
| Sensors | PASS | All N in {5,8,10} present, positions in [0,0.012] |
| Sensors.placement | WARN | Legacy sensors use block_centre_with_jitter (not podqr) |
| Metadata.conductivity | WARN | No conductivity column in metadata (BLOCKER) |
| Metadata.BC | WARN | No boundary_condition column in metadata (BLOCKER) |
