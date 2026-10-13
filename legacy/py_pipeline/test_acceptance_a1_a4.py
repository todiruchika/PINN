#!/usr/bin/env python3
"""
Acceptance Test Suite for A1, A2, A3, A4.
Verifies all 4 acceptance criteria and prints exact results.
"""
import sys
import json
import pandas as pd
import numpy as np

sys.path.insert(0, 'py_pipeline')
from validate_dataset import verify_sample_ids, ENERGY_BAL_TOL_PCT, ENERGY_METADATA_DISCREPANCY_TOL_PP


def test_a1():
    print("="*60)
    print("ACCEPTANCE TEST A1: splits.csv and metadata row counts and non-empty splits")
    print("="*60)
    df_splits = pd.read_csv("splits.csv")
    df_meta   = pd.read_csv("metadata.csv")

    n_splits = len(df_splits)
    n_meta   = len(df_meta)
    print(f"Number of rows in splits.csv:   {n_splits}")
    print(f"Number of rows in metadata.csv: {n_meta}")
    assert n_splits == n_meta, f"Row count mismatch: {n_splits} != {n_meta}"

    split_counts = df_splits["split"].value_counts().to_dict()
    print(f"Split counts by split: {split_counts}")
    for sp_name, cnt in split_counts.items():
        assert cnt > 0, f"Split '{sp_name}' is empty (count={cnt})"
        print(f"  Split '{sp_name}': {cnt} samples (non-empty: True)")

    print("RESULT: PASS")
    return True


def test_a2():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST A2: Exact sample-ID set equality and deliberate removal test")
    print("="*60)
    df_meta = pd.read_csv("metadata.csv")
    metadata_ids = set(df_meta["sample_id"])

    # Extract audited IDs from validator_full_output.txt
    with open("validator_full_output.txt") as f:
        txt = f.read()

    # Audited sample IDs from log lines: [XXX/YYY] thermal_XXXXX:
    import re
    audited_ids = set(re.findall(r'\[\d+/\d+\]\s+(thermal_\d{5}):', txt))

    print(f"Metadata sample IDs count: {len(metadata_ids)}")
    print(f"Audited sample IDs count:  {len(audited_ids)}")

    # 1. Genuine test: must report zero missing and zero extra
    check_genuine = verify_sample_ids(metadata_ids, audited_ids)
    print(f"Genuine check - missing count: {check_genuine['missing_count']}, extra count: {check_genuine['extra_count']}")
    print(f"Genuine check exact match:    {check_genuine['exact_match']}")
    assert check_genuine["missing_count"] == 0, f"Missing IDs found: {check_genuine['missing_ids']}"
    assert check_genuine["extra_count"] == 0, f"Extra IDs found: {check_genuine['extra_ids']}"
    assert check_genuine["exact_match"] is True

    # 2. Deliberately remove a row: must fail and identify the removed row
    removed_id = "thermal_00042"
    tampered_audited_ids = audited_ids - {removed_id}
    check_tampered = verify_sample_ids(metadata_ids, tampered_audited_ids)
    print(f"\nTampered check (deliberately removed '{removed_id}'):")
    print(f"  Missing count: {check_tampered['missing_count']}")
    print(f"  Missing IDs:   {check_tampered['missing_ids']}")
    print(f"  Exact match:   {check_tampered['exact_match']}")
    assert check_tampered["exact_match"] is False, "Tampered check should have failed"
    assert check_tampered["missing_ids"] == [removed_id], f"Expected [{removed_id}], got {check_tampered['missing_ids']}"

    print("RESULT: PASS")
    return True


def test_a3():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST A3: One consistent energy error per sample and threshold recount")
    print("="*60)
    with open("validation_summary.json") as f:
        summary = json.load(f)

    # Extract per-sample energy errors from validator_full_output.txt
    with open("validator_full_output.txt") as f:
        txt = f.read()

    import re
    # Match: thermal_XXXXX: ... energy_err=X.XXXX%
    matches = re.findall(r'(thermal_\d{5}):.*?energy_err=([0-9.]+)%', txt)
    print(f"Total per-sample energy error values extracted: {len(matches)}")
    assert len(matches) == 100, f"Expected 100 samples, got {len(matches)}"

    # Check exactly one consistent value per sample ID
    sample_energy_map = {}
    for sid, err_str in matches:
        err_val = float(err_str)
        assert sid not in sample_energy_map, f"Duplicate sample entry for {sid}"
        sample_energy_map[sid] = err_val

    per_sample_errors = list(sample_energy_map.values())
    max_err = max(per_sample_errors)
    med_err = float(np.median(per_sample_errors))

    print(f"Per-sample values: count = {len(per_sample_errors)}")
    print(f"Calculated from per-sample log: max = {max_err:.4f}%, median = {med_err:.4f}%")
    print(f"Summary JSON reported:         max = {summary['energy_balance']['max_error_pct']:.4f}%, median = {summary['energy_balance']['median_error_pct']:.4f}%")

    assert abs(max_err - summary['energy_balance']['max_error_pct']) < 1e-3
    assert abs(med_err - summary['energy_balance']['median_error_pct']) < 1e-3

    # Recount samples exceeding threshold 15.0%
    recount_above_15 = [sid for sid, err in sample_energy_map.items() if err > ENERGY_BAL_TOL_PCT]
    print(f"Recount exceeding {ENERGY_BAL_TOL_PCT}% threshold: count = {len(recount_above_15)}, list = {recount_above_15}")
    print(f"Summary JSON reported exceeding count:       {summary['energy_balance']['samples_above_15pct_count']}")
    assert len(recount_above_15) == summary['energy_balance']['samples_above_15pct_count']
    assert recount_above_15 == summary['energy_balance']['samples_above_15pct_list']

    print("RESULT: PASS")
    return True


def test_a4():
    print("\n" + "="*60)
    print("ACCEPTANCE TEST A4: validation_summary.json recount match against metadata.csv and splits.csv")
    print("="*60)
    with open("validation_summary.json") as f:
        summary = json.load(f)

    df_meta   = pd.read_csv("metadata.csv")
    df_splits = pd.read_csv("splits.csv")
    merged = pd.merge(df_meta, df_splits, on="sample_id", suffixes=("_meta", "_split"))

    print("Independent recount from metadata.csv and splits.csv:")
    for sp in ["train", "validation", "test"]:
        sub = merged[merged["split_split"] == sp]
        cnt = len(sub)
        above_125 = int((sub["peak_temp_C"] > 125.0).sum())
        flat = int(((sub["peak_temp_C"] - sub["min_temp_C"]) < 0.5).sum())

        json_cnt = summary["sample_counts_per_split"][sp]
        json_125 = summary["samples_above_125C_per_split"][sp]
        json_flat = summary["flat_samples_per_split"][sp]

        print(f"  Split '{sp:10s}':")
        print(f"    Sample count:       recount = {cnt:2d}, JSON = {json_cnt:2d} -> MATCH: {cnt == json_cnt}")
        print(f"    Above 125 °C:       recount = {above_125:2d}, JSON = {json_125:2d} -> MATCH: {above_125 == json_125}")
        print(f"    Flat (<0.5K):       recount = {flat:2d}, JSON = {json_flat:2d} -> MATCH: {flat == json_flat}")

        assert cnt == json_cnt, f"Split count mismatch for {sp}"
        assert above_125 == json_125, f"Above 125C count mismatch for {sp}"
        assert flat == json_flat, f"Flat count mismatch for {sp}"

    total_cnt = len(merged)
    total_125 = int((merged["peak_temp_C"] > 125.0).sum())
    total_flat = int(((merged["peak_temp_C"] - merged["min_temp_C"]) < 0.5).sum())
    assert total_cnt == summary["sample_counts_per_split"]["total"]
    assert total_125 == summary["samples_above_125C_per_split"]["total"]
    assert total_flat == summary["flat_samples_per_split"]["total"]

    print(f"  Totals: samples={total_cnt}, above_125C={total_125}, flat={total_flat}")
    print("RESULT: PASS")
    return True


if __name__ == "__main__":
    t1 = test_a1()
    t2 = test_a2()
    t3 = test_a3()
    t4 = test_a4()
    print("\n" + "="*60)
    print("ALL 4 ACCEPTANCE TESTS (A1, A2, A3, A4): PASS")
    print("="*60)
