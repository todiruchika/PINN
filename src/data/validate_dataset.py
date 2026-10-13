"""Dataset validation CLI.

Usage: python -m src.data.validate_dataset --dir data/raw/dataset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DIE_L = 0.012
GRID = 128
SENSOR_COUNTS = [5, 8, 10]


def _edge_pixels(T: np.ndarray) -> np.ndarray:
    top = T[0, :]
    bottom = T[-1, :]
    left = T[1:-1, 0]
    right = T[1:-1, -1]
    return np.concatenate([top, bottom, left, right])


def run_validation(data_dir: Path):
    results = []
    fail = False

    def record(name, status, msg):
        nonlocal fail
        tag = f"[{status}]"
        print(f"{tag:<8} {name}: {msg}")
        results.append({"check": name, "status": status, "message": msg})
        if status == "FAIL":
            fail = True

    # 1. Layout
    splits = ["train", "validation", "test"]
    for split in splits:
        for subdir in ["matrices", "images", "sensors"]:
            p = data_dir / split / subdir
            if not p.exists():
                record("Layout", "FAIL", f"Missing: {p}")
            else:
                record("Layout", "PASS", f"{p} exists")

    meta = pd.read_csv(data_dir / "metadata.csv")
    splits_df = pd.read_csv(data_dir / "splits.csv")

    meta_ids = set(meta["sample_id"])
    splits_ids = set(splits_df["sample_id"])
    if meta_ids == splits_ids:
        record("Layout.IDs", "PASS", f"{len(meta_ids)} IDs consistent across metadata and splits.csv")
    else:
        only_meta = meta_ids - splits_ids
        only_splits = splits_ids - meta_ids
        record("Layout.IDs", "FAIL", f"Mismatch: only in metadata={len(only_meta)}, only in splits.csv={len(only_splits)}")

    for split in splits:
        matrix_ids = {f.stem for f in (data_dir / split / "matrices").glob("thermal_?????.npy") if not f.stem.endswith("_Q")}
        expected = set(meta.loc[meta["split"] == split, "sample_id"])
        if matrix_ids == expected:
            record(f"Layout.{split}.matrix_ids", "PASS", f"{len(matrix_ids)} IDs match")
        else:
            record(f"Layout.{split}.matrix_ids", "FAIL", f"Expected {len(expected)}, got {len(matrix_ids)}")

    # 2. Splits
    split_counts = splits_df["split"].value_counts().to_dict()
    record("Splits.counts", "PASS",
           f"train={split_counts.get('train',0)}, validation={split_counts.get('validation',0)}, test={split_counts.get('test',0)}")

    overlaps = []
    for s1 in splits:
        for s2 in splits:
            if s1 < s2:
                s1_ids = set(splits_df.loc[splits_df["split"] == s1, "sample_id"])
                s2_ids = set(splits_df.loc[splits_df["split"] == s2, "sample_id"])
                if s1_ids & s2_ids:
                    overlaps.append(f"{s1}&{s2}: {len(s1_ids & s2_ids)}")
    if overlaps:
        record("Splits.overlap", "FAIL", "; ".join(overlaps))
    else:
        record("Splits.overlap", "PASS", "No overlap across splits")

    for split in splits:
        split_meta = pd.read_csv(data_dir / split / "metadata.csv")
        csv_ids = set(splits_df.loc[splits_df["split"] == split, "sample_id"])
        split_meta_ids = set(split_meta["sample_id"])
        if csv_ids == split_meta_ids:
            record(f"Splits.{split}.meta_agree", "PASS", f"{len(csv_ids)} IDs agree")
        else:
            record(f"Splits.{split}.meta_agree", "FAIL", "splits.csv vs per-split metadata mismatch")

    # 3. Grids
    grid_issues = []
    for split in splits:
        for npy in sorted((data_dir / split / "matrices").glob("*.npy")):
            arr = np.load(str(npy))
            if arr.shape != (GRID, GRID):
                grid_issues.append(f"{npy.name}: shape {arr.shape}")
            if not np.isfinite(arr).all():
                grid_issues.append(f"{npy.name}: NaN/Inf")
            if "_Q" in npy.name and (arr < 0).any():
                grid_issues.append(f"{npy.name}: Q<0")
    if grid_issues:
        record("Grids", "FAIL", "; ".join(grid_issues[:5]))
    else:
        record("Grids", "PASS", "All grids (128,128), finite, Q>=0")

    # 4. Temperature
    peak_temps = meta["peak_temp_C"].values
    above_125 = int((peak_temps > 125).sum())
    record("Temperature", "PASS",
           f"peak_T: min={peak_temps.min():.1f} median={np.median(peak_temps):.1f} max={peak_temps.max():.1f} degC; "
           f"{above_125} samples > 125 degC")

    # 5. Boundary (BLOCKER — expected to FAIL)
    max_deviations = []
    for split in splits:
        for sid in meta.loc[meta["split"] == split, "sample_id"]:
            T_amb = float(meta.loc[meta["sample_id"] == sid, "T_amb_C"].values[0])
            T = np.load(str(data_dir / split / "matrices" / f"{sid}.npy"))
            edge = _edge_pixels(T)
            dev = np.abs(edge - T_amb)
            max_deviations.append(dev.max())
    overall_max = float(np.max(max_deviations))
    overall_mean = float(np.mean(max_deviations))
    if overall_max > 0.5:
        record("Boundary", "FAIL",
               f"BLOCKER: max edge deviation from T_amb = {overall_max:.2f} degC "
               f"(mean {overall_mean:.2f} degC). BC is unknown.")
    else:
        record("Boundary", "PASS", f"max edge deviation = {overall_max:.4f} degC")

    # 6. Energy
    dA = (DIE_L / GRID) ** 2
    energy_errors = []
    for split in splits:
        for sid in meta.loc[meta["split"] == split, "sample_id"]:
            P_in = float(meta.loc[meta["sample_id"] == sid, "energy_bal_P_in"].values[0])
            Q = np.load(str(data_dir / split / "matrices" / f"{sid}_Q.npy"))
            ratio = (Q.sum() * dA) / P_in
            energy_errors.append(abs(ratio - 1))

    max_err = float(np.max(energy_errors))
    median_err = float(np.median(energy_errors))
    if max_err > 0.03:
        record("Energy", "FAIL", f"max |ratio-1| = {max_err:.4f} > 0.03")
    else:
        record("Energy", "PASS", f"max |ratio-1| = {max_err:.4f}, median = {median_err:.4f}")

    err_pct_max = float(meta["energy_bal_err_pct"].max())
    if err_pct_max > 2.5:
        record("Energy.err_pct", "WARN", f"energy_bal_err_pct max = {err_pct_max:.2f}% > 2.5%")
    else:
        record("Energy.err_pct", "PASS", f"energy_bal_err_pct max = {err_pct_max:.2f}%")

    # 7. Sensors
    sensor_missing = []
    sensor_pos_issues = []
    sensor_quant_warnings = []
    for split in splits:
        for sid in meta.loc[meta["split"] == split, "sample_id"]:
            for N in SENSOR_COUNTS:
                p = data_dir / split / "sensors" / f"{sid}_sensors_{N}.json"
                if not p.exists():
                    sensor_missing.append(p.name)
                    continue
                with open(p) as f:
                    s = json.load(f)
                for pos in s["positions"]:
                    if not (0 <= pos[0] <= DIE_L and 0 <= pos[1] <= DIE_L):
                        sensor_pos_issues.append(f"{p.name}: pos {pos}")
                # Check noisy values are integer-quantized (noisy % 1.0 == 0)
                for t_noisy in s["temperatures_noisy"]:
                    if abs(t_noisy - round(t_noisy)) > 0.01:
                        sensor_quant_warnings.append(p.name)
                        break
    if sensor_missing:
        record("Sensors", "FAIL", f"Missing {len(sensor_missing)} files: " + "; ".join(sensor_missing[:3]))
    elif sensor_pos_issues:
        record("Sensors", "FAIL", f"Position out of range: " + "; ".join(sensor_pos_issues[:3]))
    else:
        record("Sensors", "PASS", f"All N in {{5,8,10}} present, positions in [0,{DIE_L}]")
    if sensor_quant_warnings:
        record("Sensors.quantization", "WARN",
               f"{len(sensor_quant_warnings)} sensor files have non-integer noisy values (expected for legacy sensors with continuous true values)")
    record("Sensors.placement", "WARN", "Legacy sensors use block_centre_with_jitter (not podqr)")

    # 8. Metadata
    if "k_thermal" not in meta.columns and "conductivity" not in meta.columns:
        record("Metadata.conductivity", "WARN", "No conductivity column in metadata (BLOCKER)")
    else:
        record("Metadata.conductivity", "PASS", "Conductivity column present")
    if "boundary_condition" not in meta.columns:
        record("Metadata.BC", "WARN", "No boundary_condition column in metadata (BLOCKER)")
    else:
        record("Metadata.BC", "PASS", "Boundary condition column present")

    return results, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw/dataset", type=Path)
    args = parser.parse_args()

    results, has_fail = run_validation(args.dir)

    out_dir = Path("results/validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    report_md = "\n".join(
        f"| {r['check']} | {r['status']} | {r['message']} |" for r in results
    )
    header = "| Check | Status | Message |\n|-------|--------|---------|"
    (out_dir / "dataset_report.md").write_text(
        f"# Dataset Validation Report\n\n{header}\n{report_md}\n"
    )
    with open(out_dir / "dataset_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport written to {out_dir}/dataset_report.{{md,json}}")
    sys.exit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
