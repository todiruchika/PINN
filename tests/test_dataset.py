import numpy as np
import pytest
from src.data.dataset import load_sample, load_metadata

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


def test_all_100_samples():
    meta = get_meta()
    assert len(meta) == 100
    for _, row in meta.iterrows():
        for n in ALL_NS:
            result = load_sample(row["sample_id"], n, strategy="block_centre_with_jitter")
            assert result["T"].shape == (128, 128)
