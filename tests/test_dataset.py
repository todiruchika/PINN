import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.data.dataset import load_sample, load_metadata
from src.sensors.layouts import podqr_positions_m

ALL_NS = [5, 8, 10]


def get_meta():
    return load_metadata()


def test_loader_shape():
    meta = get_meta()
    for split in ["train", "validation", "test"]:
        ids = meta.loc[meta["split"] == split, "sample_id"].tolist()[:1]
        for sid in ids:
            for n in ALL_NS:
                result = load_sample(sid, n, strategy="block_centre_with_jitter")
                assert result["T"].shape == (128, 128), f"{sid} T shape wrong"
                assert result["Q"].shape == (128, 128), f"{sid} Q shape wrong"
                assert result["sensors"]["sensor_count"] == n
                assert len(result["sensors"]["positions"]) == n


def test_all_samples():
    meta = get_meta()
    assert len(meta) == 700
    for _, row in meta.iterrows():
        for n in ALL_NS:
            result = load_sample(row["sample_id"], n, strategy="block_centre_with_jitter")
            assert result["T"].shape == (128, 128)


def _temporary_sample(tmp_path):
    raw_dir = tmp_path / "raw"
    sample_id = "thermal_00001"
    matrix_dir = raw_dir / "train" / "matrices"
    sensor_dir = raw_dir / "train" / "sensors"
    matrix_dir.mkdir(parents=True)
    sensor_dir.mkdir(parents=True)
    temperature = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)
    np.save(matrix_dir / f"{sample_id}.npy", temperature)
    np.save(matrix_dir / f"{sample_id}_Q.npy", np.ones((128, 128), dtype=np.float32))
    metadata = pd.DataFrame([{"sample_id": sample_id, "split": "train", "T_amb_C": 25.0}])
    metadata.to_csv(raw_dir / "metadata.csv", index=False)
    reference_record = {
        "sensor_count": 5,
        "positions": [[0.003 + i * 0.001, 0.004] for i in range(5)],
        "temperatures_true": [30.0] * 5,
        "temperatures_noisy": [30.0] * 5,
        "placement_strategy": "block_centre_with_jitter",
    }
    with open(sensor_dir / f"{sample_id}_sensors_5.json", "w") as f:
        json.dump(reference_record, f)
    return raw_dir, sample_id, metadata


def test_random_uniform_strategy_returns_unique_random_layout(tmp_path):
    raw_dir, sample_id, metadata = _temporary_sample(tmp_path)
    first = load_sample(sample_id, 5, "random_uniform", raw_dir, metadata)
    second = load_sample(sample_id, 5, "random_uniform", raw_dir, metadata)

    assert first["sensors"]["placement_strategy"] == "random_uniform"
    assert len(set(map(tuple, first["sensors"]["positions"]))) == 5
    assert first["sensors"] == second["sensors"]


def test_podqr_missing_file_does_not_fall_back_to_reference(tmp_path):
    raw_dir, sample_id, metadata = _temporary_sample(tmp_path)

    with pytest.raises(FileNotFoundError, match="Generate them"):
        load_sample(
            sample_id,
            5,
            "podqr",
            raw_dir,
            metadata,
            generated_dir=tmp_path / "missing_generated",
        )


def test_podqr_strategy_requires_podqr_record(tmp_path):
    raw_dir, sample_id, metadata = _temporary_sample(tmp_path)
    generated_dir = tmp_path / "generated"
    sensor_dir = generated_dir / "train"
    sensor_dir.mkdir(parents=True)
    record = {
        "sensor_count": 5,
        "positions": podqr_positions_m(5),
        "temperatures_true": [30.0] * 5,
        "temperatures_noisy": [30.0] * 5,
        "placement_strategy": "podqr",
    }
    with open(sensor_dir / f"{sample_id}_sensors_5.json", "w") as f:
        json.dump(record, f)

    result = load_sample(
        sample_id, 5, "podqr", raw_dir, metadata, generated_dir=generated_dir
    )
    assert result["sensors"]["placement_strategy"] == "podqr"


def test_podqr_stale_layout_fails_instead_of_loading(tmp_path):
    raw_dir, sample_id, metadata = _temporary_sample(tmp_path)
    generated_dir = tmp_path / "generated"
    sensor_dir = generated_dir / "train"
    sensor_dir.mkdir(parents=True)
    record = {
        "sensor_count": 5,
        "positions": [[0.005, 0.005]] * 5,
        "temperatures_true": [30.0] * 5,
        "temperatures_noisy": [30.0] * 5,
        "placement_strategy": "podqr",
    }
    with open(sensor_dir / f"{sample_id}_sensors_5.json", "w") as f:
        json.dump(record, f)

    with pytest.raises(ValueError, match="stale or does not match"):
        load_sample(
            sample_id, 5, "podqr", raw_dir, metadata, generated_dir=generated_dir
        )


def test_unknown_sensor_strategy_fails_instead_of_falling_back(tmp_path):
    raw_dir, sample_id, metadata = _temporary_sample(tmp_path)
    with pytest.raises(ValueError, match="Unknown sensor strategy"):
        load_sample(sample_id, 5, "typo", raw_dir, metadata)
