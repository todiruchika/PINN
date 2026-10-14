import pandas as pd
import pytest

from src.eval.coverage import (
    COLUMNS,
    build_report,
    count_matrices,
    load_extended_metadata,
    summarize,
)


def _meta(peaks, amb, blocks, viol):
    return pd.DataFrame({
        "peak_temp_C": peaks,
        "T_amb_C": amb,
        "block_count": blocks,
        "thermal_violation": viol,
    })


def test_summarize_known_values():
    df = _meta([10.0, 20.0, 30.0], [25.0, 35.0, 45.0], [2, 4, 6], [False, True, False])
    s = summarize(df)
    assert s["peak_temp_C"]["min"] == pytest.approx(10.0)
    assert s["peak_temp_C"]["median"] == pytest.approx(20.0)
    assert s["peak_temp_C"]["max"] == pytest.approx(30.0)
    assert s["T_amb_C"]["mean"] == pytest.approx(35.0)
    assert s["block_count"]["n"] == 3
    assert s["thermal_violation"]["mean"] == pytest.approx(1 / 3)


def test_report_marks_missing_extended():
    delivered = _meta([10.0, 20.0], [25.0, 30.0], [2, 3], [False, False])
    text = build_report(delivered, None, {"train": 5, "val": 0, "test": 0})
    assert "Extended metadata not available" in text
    assert "| train | 5 |" in text


def test_report_coverage_flag():
    delivered = _meta([10.0, 50.0], [25.0, 40.0], [2, 8], [False, True])
    wide = _meta([5.0, 60.0], [24.0, 46.0], [1, 9], [False, True])
    narrow = _meta([20.0, 30.0], [30.0, 35.0], [3, 4], [False, False])

    wide_text = build_report(delivered, wide, {})
    narrow_text = build_report(delivered, narrow, {})

    peak_row = next(l for l in wide_text.splitlines() if l.startswith("| peak_temp_C | extended"))
    assert peak_row.rstrip().endswith("yes |")
    peak_row = next(l for l in narrow_text.splitlines() if l.startswith("| peak_temp_C | extended"))
    assert peak_row.rstrip().endswith("no |")


def test_load_extended_metadata_missing(tmp_path):
    assert load_extended_metadata(tmp_path / "absent") is None
    (tmp_path / "train").mkdir()
    assert load_extended_metadata(tmp_path) is None


def test_load_extended_metadata_concatenates_splits(tmp_path):
    for split, n in [("train", 2), ("val", 1)]:
        d = tmp_path / split
        d.mkdir()
        _meta([1.0] * n, [30.0] * n, [3] * n, [False] * n).to_csv(d / "metadata.csv", index=False)
    df = load_extended_metadata(tmp_path)
    assert len(df) == 3
    assert list(df.columns) == COLUMNS


def test_count_matrices_excludes_q(tmp_path):
    mats = tmp_path / "train" / "matrices"
    mats.mkdir(parents=True)
    for name in ["thermal_00001.npy", "thermal_00001_Q.npy", "thermal_00002.npy"]:
        (mats / name).write_bytes(b"")
    assert count_matrices(tmp_path)["train"] == 2
    assert count_matrices(tmp_path)["val"] == 0
