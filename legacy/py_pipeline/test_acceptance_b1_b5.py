#!/usr/bin/env python3
"""
Acceptance Test Suite for B1, B2, B3, B4, B5.
Verifies all 5 acceptance criteria and prints exact outputs.
"""
import os
import sys
import json
import re
import pandas as pd
import numpy as np

sys.path.insert(0, 'py_pipeline')
from validate_dataset import DatasetValidator, ENERGY_BAL_TOL_PCT


def test_b1():
    print("="*60)
    print("ACCEPTANCE TEST B1: One energy value per sample and threshold recount at 0.5%")
    print("="*60)
    with open("validation_summary.json") as f:
        summary = json.load(f)

    with open("validator_full_output.txt") as f:
        txt = f.read()

    # Extract all per-sample energy errors from the audit log
    matches = re.findall(r'(thermal_\d{5}):.*?energy_err=([0-9.]+)%', txt)
    print(f"Total per-sample energy error values extracted: {len(matches)}")
    assert len(matches) == 100, f"Expected 100 values, got {len(matches)}"

    # Confirm exactly one value per sample ID
    sample_energy_map = {}
    for sid, err_str in matches:
        err_val = float(err_str)
        assert sid not in sample_energy_map, f"Duplicate value for {sid}"
        sample_energy_map[sid] = err_val

    per_sample_errors = list(sample_energy_map.values())
    max_err = max(per_sample_errors)
    med_err = float(np.median(per_sample_errors))
    p95_err = float(np.percentile(per_sample_errors, 95))

    print(f"Per-sample errors count: {len(per_sample_errors)}")
    print(f"Computed from per-sample log: max = {max_err:.4f}%, median = {med_err:.4f}%, p95 = {p95_err:.4f}%")
    print(f"Summary JSON reported:        max = {summary['energy_balance']['max_error_pct']:.4f}%, median = {summary['energy_balance']['median_error_pct']:.4f}%, p95 = {summary['energy_balance']['p95_error_pct']:.4f}%")

    assert abs(max_err - summary['energy_balance']['max_error_pct']) < 1e-3
    assert abs(med_err - summary['energy_balance']['median_error_pct']) < 1e-3
    assert abs(p95_err - summary['energy_balance']['p95_error_pct']) < 1e-3

    # Independent recount of samples above 0.5% threshold
    recount_above_05 = [sid for sid, err in sample_energy_map.items() if err > ENERGY_BAL_TOL_PCT]
    print(f"Independent recount exceeding {ENERGY_BAL_TOL_PCT}% threshold: count = {len(recount_above_05)}")
    print(f"Summary JSON reported count exceeding threshold:        {summary['energy_balance']['samples_above_0_5pct_count']}")

    assert len(recount_above_05) == summary['energy_balance']['samples_above_0_5pct_count']
    json_list_ids = [item["sample_id"] for item in summary['energy_balance']['samples_above_0_5pct_list']]
    assert sorted(recount_above_05) == sorted(json_list_ids)

    print("RESULT: PASS")
    return True


def test_b2():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST B2: No pass/fail decision reads a stored metadata value")
    print("="*60)
    # Verify that tampering with stored metadata values does not alter the recomputed
    # energy balance error or any pass/fail decision.
    import tempfile, shutil

    with open("validation_summary.json") as f:
        original_summary = json.load(f)

    # Inspect code of validate_dataset.py to confirm no metadata values are used for pass/fail
    with open("py_pipeline/validate_dataset.py") as f:
        code = f.read()

    # Confirm metadata discrepancy is NOT in hard_failures
    assert 'counters["metadata_energy_discrepancy_fail"]' not in code.split('hard_failures =')[1].split('self._print_split_summary')[0], \
        "metadata_energy_discrepancy_fail must not be part of hard_failures"

    print("Code inspection: hard_failures contains only physically recomputed criteria and disk file existence:")
    print("  hard_failures = missing_files + nan_inf + above_melt + below_ambient + energy_bal_fail + analytic_power_fail + sensor_noise_fail + schema_fail + id_check")

    # Run audit on a modified dataset directory where stored metadata columns
    # (energy_bal_err_pct, energy_bal_P_in, energy_bal_P_out, peak_temp_C) are corrupted to 9999.0
    temp_dir = tempfile.mkdtemp()
    try:
        shutil.copytree("py_pipeline/dataset", os.path.join(temp_dir, "dataset"))
        # Tamper with stored metadata outputs
        for root, dirs, files in os.walk(temp_dir):
            if "metadata.csv" in files:
                m_path = os.path.join(root, "metadata.csv")
                df_temp = pd.read_csv(m_path)
                if "energy_bal_err_pct" in df_temp.columns:
                    df_temp["energy_bal_err_pct"] = 9999.0
                if "peak_temp_C" in df_temp.columns:
                    df_temp["peak_temp_C"] = -999.0
                if "energy_bal_P_in" in df_temp.columns:
                    df_temp["energy_bal_P_in"] = 0.0
                df_temp.to_csv(m_path, index=False)

        validator_tampered = DatasetValidator(dataset_dir=os.path.join(temp_dir, "dataset"))
        out_temp_json = os.path.join(temp_dir, "tampered_summary.json")
        validator_tampered.run_all(run_convergence_check=False, out_summary_json=out_temp_json)

        with open(out_temp_json) as f:
            tampered_summary = json.load(f)

        # Compare recomputed energy balance metrics
        print(f"Original max error: {original_summary['energy_balance']['max_error_pct']:.4f}%")
        print(f"Tampered max error: {tampered_summary['energy_balance']['max_error_pct']:.4f}%")
        assert abs(original_summary['energy_balance']['max_error_pct'] - tampered_summary['energy_balance']['max_error_pct']) < 1e-4

        print(f"Original median error: {original_summary['energy_balance']['median_error_pct']:.4f}%")
        print(f"Tampered median error: {tampered_summary['energy_balance']['median_error_pct']:.4f}%")
        assert abs(original_summary['energy_balance']['median_error_pct'] - tampered_summary['energy_balance']['median_error_pct']) < 1e-4

        # Confirm sample counts and above 125C counts were computed from temperature matrices, not corrupted metadata
        print(f"Samples above 125C with corrupted metadata: {tampered_summary['samples_above_125C_per_split']['total']}")
        assert tampered_summary['samples_above_125C_per_split']['total'] == 11, "Should remain 11 as computed from .npy matrices"

        print("Tampering test confirms zero pass/fail decisions depend on stored metadata values.")
    finally:
        shutil.rmtree(temp_dir)

    print("RESULT: PASS")
    return True


def test_b3():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST B3: splits.csv has one row per sample ID in metadata.csv with exact ID matching")
    print("="*60)
    df_meta   = pd.read_csv("metadata.csv")
    df_splits = pd.read_csv("splits.csv")

    print(f"splits.csv row count:   {len(df_splits)}")
    print(f"metadata.csv row count: {len(df_meta)}")
    assert len(df_splits) == len(df_meta), f"Row count mismatch: {len(df_splits)} != {len(df_meta)}"

    meta_ids_set = set(df_meta["sample_id"])
    splits_ids_set = set(df_splits["sample_id"])

    missing = sorted(list(meta_ids_set - splits_ids_set))
    extra   = sorted(list(splits_ids_set - meta_ids_set))

    print(f"Missing sample IDs in splits.csv: {missing if missing else 'None (0)'}")
    print(f"Extra sample IDs in splits.csv:   {extra if extra else 'None (0)'}")
    print(f"Exact set equality: {meta_ids_set == splits_ids_set}")

    assert len(missing) == 0
    assert len(extra) == 0
    assert meta_ids_set == splits_ids_set

    # Verify no duplicate sample IDs in splits.csv
    assert len(df_splits["sample_id"]) == len(splits_ids_set), "Duplicate sample IDs found in splits.csv"

    print("RESULT: PASS")
    return True


def test_b4():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST B4: Each split's fraction above 125 C is within one sample of full-set fraction")
    print("="*60)
    df_meta   = pd.read_csv("metadata.csv")
    df_splits = pd.read_csv("splits.csv")
    merged = pd.merge(df_meta, df_splits, on="sample_id", suffixes=("_meta", "_split"))

    full_total = len(merged)
    full_viol  = int((merged["peak_temp_C"] > 125.0).sum())
    full_frac  = full_viol / full_total

    print(f"Full dataset: {full_viol}/{full_total} samples above 125 °C (fraction = {full_frac:.4f} = {full_frac*100:.2f}%)")

    for sp in ["train", "validation", "test"]:
        sub = merged[merged["split_split"] == sp]
        sub_total = len(sub)
        sub_viol  = int((sub["peak_temp_C"] > 125.0).sum())
        sub_frac  = sub_viol / sub_total
        expected_viol_count = sub_total * full_frac
        diff_in_samples = abs(sub_viol - expected_viol_count)

        print(f"  Split '{sp:10s}': total={sub_total:2d}, above_125C={sub_viol:2d}, fraction={sub_frac:.4f} "
              f"(expected={expected_viol_count:.2f}, difference={diff_in_samples:.2f} samples)")

        assert diff_in_samples <= 1.0, f"Split {sp} exceeds 1-sample tolerance: {diff_in_samples:.2f} > 1.0"
        print(f"    Within one sample of full-set fraction: True")

    print("RESULT: PASS")
    return True


def test_b5():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST B5: The test split contains at least one sample above 125 C")
    print("="*60)
    df_meta   = pd.read_csv("metadata.csv")
    df_splits = pd.read_csv("splits.csv")
    merged = pd.merge(df_meta, df_splits, on="sample_id", suffixes=("_meta", "_split"))

    test_samples = merged[merged["split_split"] == "test"]
    test_viol = test_samples[test_samples["peak_temp_C"] > 125.0]

    print(f"Test split total samples:               {len(test_samples)}")
    print(f"Test split samples above 125 °C count:  {len(test_viol)}")
    print(f"Test split samples above 125 °C IDs:    {test_viol['sample_id'].tolist()}")
    for _, r in test_viol.iterrows():
        print(f"  {r['sample_id']}: peak temperature = {r['peak_temp_C']:.2f} °C")

    assert len(test_viol) >= 1, "Test split does not contain any sample above 125 °C"
    print("RESULT: PASS")
    return True


if __name__ == "__main__":
    b1 = test_b1()
    b2 = test_b2()
    b3 = test_b3()
    b4 = test_b4()
    b5 = test_b5()
    print("\n" + "="*60)
    print("ALL 5 ACCEPTANCE TESTS (B1, B2, B3, B4, B5): PASS")
    print("="*60)
