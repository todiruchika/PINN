"""Small shared helpers: seeding, timing, config loading."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]          # pure_pinn/
PROJECT_ROOT = REPO_ROOT.parent                          # repository root (holds data/)


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


@contextmanager
def timer():
    """``with timer() as t: ...`` then read ``t['seconds']``."""
    record: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - start


def resolve_path(path: str | os.PathLike) -> Path:
    """Interpret relative paths against ``pure_pinn/``, not the cwd."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def resolve_project_path(path: str | os.PathLike) -> Path:
    """Interpret relative paths against the repository root (where ``data/`` lives)."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
