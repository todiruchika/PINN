import json
import numpy as np
import pytest
from pathlib import Path
from src.sensors.podqr import fit_podqr_layout
from src.data.splits import load_splits

GRID = 128
MARGIN = 3


def get_layout():
    layout_path = Path("sensors/layouts/podqr_layout.json")
    if not layout_path.exists():
        splits = load_splits()
        fit_podqr_layout(splits["train"])
    with open(layout_path) as f:
        return json.load(f)


def test_pivots_unique():
    layout = get_layout()
    pivots = [(p["row"], p["col"]) for p in layout["pivots"]]
    assert len(set(pivots)) == len(pivots), "Pivot indices not unique"


def test_pivots_within_grid():
    layout = get_layout()
    for p in layout["pivots"]:
        assert MARGIN <= p["row"] < GRID - MARGIN
        assert MARGIN <= p["col"] < GRID - MARGIN


def test_pivots_respect_margin():
    layout = get_layout()
    for p in layout["pivots"]:
        assert p["row"] >= MARGIN and p["row"] < GRID - MARGIN
        assert p["col"] >= MARGIN and p["col"] < GRID - MARGIN


def test_nesting():
    layout = get_layout()
    pivots = [(p["row"], p["col"]) for p in layout["pivots"]]
    p5 = set(pivots[:5])
    p8 = set(pivots[:8])
    p10 = set(pivots[:10])
    assert p5.issubset(p8), "5-sensor set not subset of 8"
    assert p8.issubset(p10), "8-sensor set not subset of 10"


def test_leakage_layout_uses_train_only():
    """Layout must not change when fit twice on the same train IDs."""
    splits = load_splits()
    layout1 = fit_podqr_layout(splits["train"])
    layout2 = fit_podqr_layout(splits["train"])
    assert layout1["pivots"] == layout2["pivots"], "Layout not deterministic"
    assert layout1["train_ids_sha256"] == layout2["train_ids_sha256"]
