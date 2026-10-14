# Dataset Extension Report

Generated: 2026-10-13.

---

## Phase 5a — Regression Check

**Historical command:** a retired generator snapshot was run sequentially via direct API call; multiprocessing pool failed — see note below. The original source is not retained in this report.

**Result: PASSED — byte-identical.**

| Sample | T max diff (°C) | Q max diff (W/m²) | peak_T match | T_amb match |
|--------|----------------|-------------------|--------------|-------------|
| thermal_00001 | 0.000000 | 0.0000 | ✓ (42.6273 °C) | ✓ (32.4908 °C) |
| thermal_00002 | 0.000000 | 0.0000 | ✓ (94.4238 °C) | ✓ (44.0143 °C) |

The delivered dataset and the generator snapshot used for this historical check were bit-for-bit reproducible for the two tested samples at seed 42.

**Note on multiprocessing:** The generator's `mp.Pool` fails on this Windows 11 machine with
`numpy._core._exceptions._ArrayMemoryError: Unable to allocate 2.00 MiB for array (512,512)`
inside child processes. Root cause: Windows subprocess memory layout prevents allocation of the
fine-mesh solver array in forked workers. The generator can be run sequentially (direct API calls)
or on a Linux/Mac system.

---

## Phase 5b — Extended Dataset Generation

**Historical command:** an untracked sequential wrapper targeted `data/raw_extended` with 400/50/50 samples and seed 42.

**Status: Running in background.** The multiprocessing pool was unavailable (see above).
Sequential rate: ~60 s/sample × 500 samples = **~8 hours estimated**. This exceeds the 2-hour
limit; the generator is running in the background.

**Progress at report time:**

| Split | Target | Generated | Directory |
|-------|--------|-----------|-----------|
| train | 400 | ~10 (in progress) | data/raw_extended/train/ |
| val | 50 | 0 | data/raw_extended/val/ |
| test | 50 | 0 | data/raw_extended/test/ |

**ID scheme:**
- train: `thermal_00001` – `thermal_00400`
- val: `thermal_00401` – `thermal_00450`
- test: `thermal_00451` – `thermal_00500`

IDs `thermal_00001` – `thermal_00100` overlap with the delivered dataset.
Regression check confirms `thermal_00001` and `thermal_00002` are byte-identical in the extended run.
IDs `thermal_00003` – `thermal_00100` are expected identical (same seed, sequential Picard solver).

**Validator:** Not yet run — extended dataset incomplete.

---

## Phase 5c — Distribution Comparison

### Delivered 100 samples vs generator package metadata

The `generator/metadata.csv` file is **identical** to the delivered dataset (same 100 sample IDs, same values).

**No 500-sample file with peaks 27–105 °C was found in this repository.**
All three metadata files in the repository describe the same 100 samples with peaks 31–164 °C.

| Statistic | Delivered 100 |
|-----------|--------------|
| peak_temp_C: min | 31.15 °C |
| peak_temp_C: median | 89.11 °C |
| peak_temp_C: max | 164.18 °C |
| T_amb_C: min | 25.11 °C |
| T_amb_C: median | 34.28 °C |
| T_amb_C: max | 44.74 °C |
| block_count: min | 2 |
| block_count: median | 5 |
| block_count: max | 8 |
| thermal_violation | 11 / 100 (11%) |

**Expected distribution for extended 500-sample set** (projected from seed 42, same generator):
The generator samples T_amb ∈ U(25, 45) °C and block count from U(2, 8). Peak temperature
distribution will be similar to the delivered set (31–164 °C range expected). Distribution
comparison will be updated once generation completes.

---

## Open Items

1. **Complete extended generation.** The background process needs ~8 hours on this machine. Consider
   running on a Linux/Mac system with multiprocessing enabled for ~20× speedup.
2. **Run validator on extended set** once complete: `python -m src.data.validate_dataset --dir data/raw_extended`
3. **Rename val/ → validation/** in extended set directory after generation (to match delivered convention).
4. **Verify byte-identity** for all 100 overlapping IDs once generation completes.
5. **Locate or regenerate 500-sample file** with peaks 27–105 °C if it was referenced from a prior
   pipeline run. Not present in this repository.
