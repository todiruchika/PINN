"""Real commercial-processor thermal maps (Lu and Tan, MLCAD 2024).

Source: https://github.com/sheldonucr/commercial_thermal_map_dataset  (no license is stated there; the data is NOT
redistributed in this repository, download it with ``python -m src.data.real_thermal --download``).

Each ``*.pkl`` holds ``{"input": performance-counter features (n, f), "output": thermal maps (n, H, W) in degC}``.
Pickle can execute code when loaded, so files are read only through :func:`safe_load`, which accepts nothing but
NumPy array reconstruction and refuses every other import.

Splits are made so that no test frame has a near-identical neighbour in training (consecutive frames of a run are
almost the same image): whole workloads for processors that have several (i5, i7), contiguous time blocks separated
by gaps for processors with one recording (Ryzen, RTX 4060, TPU).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np

RAW_URL = "https://raw.githubusercontent.com/sheldonucr/commercial_thermal_map_dataset/main/data_files/"
DATA_DIR = Path("data/real_thermal")
MANIFEST = Path("configs/real_thermal_sha256.txt")

I5 = ["7zip", "aobench", "flac", "opencv", "stream"]
I7 = ["aio-stress", "aobench", "build-gcc", "cachebench", "compress-7zip", "cyclictest", "dbench", "encode-flac", "fio",
      "gimp", "git", "idle", "mbw", "phpbench", "ramspeed", "stream", "t-test1", "tinymembench", "tiobench"]
PROCESSORS = {
    "i5": [f"CPU_i5_{w}.pkl" for w in I5],
    "i7": [f"CPU_i7_{w}.pkl" for w in I7],
    "r7_4800u": ["CPU_R7_4800U.pkl"],
    "rtx4060": ["GPU_RTX_4060.pkl"],
    "tpu": ["TPU_Google_Edge.pkl"],
}

_ALLOWED = {
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
}


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked import {module}.{name} (only NumPy arrays are allowed)")


def safe_load(path: Path | str) -> dict:
    """Load a dataset pickle, refusing anything that is not a plain dict of NumPy arrays."""
    with open(path, "rb") as f:
        obj = _SafeUnpickler(io.BytesIO(f.read())).load()
    if not isinstance(obj, dict) or set(obj) != {"input", "output"}:
        raise ValueError(f"{path}: expected a dict with keys 'input' and 'output'")
    out = np.asarray(obj["output"])
    if out.ndim != 3:
        raise ValueError(f"{path}: expected thermal maps of shape (n, H, W), got {out.shape}")
    return {"input": np.asarray(obj["input"]), "output": out.astype(np.float32)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return {ln.split()[1]: ln.split()[0] for ln in MANIFEST.read_text().splitlines() if ln.strip()}


def download(processors=None, data_dir: Path = DATA_DIR) -> None:
    """Fetch the files (skipping ones already present) and verify them against the manifest if there is one."""
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    new = {}
    for proc in processors or PROCESSORS:
        for fname in PROCESSORS[proc]:
            dest = data_dir / fname
            if not dest.exists():
                print(f"downloading {fname} ...", flush=True)
                urllib.request.urlretrieve(RAW_URL + fname, dest)
            digest = sha256(dest)
            if fname in manifest and manifest[fname] != digest:
                raise RuntimeError(f"{fname}: checksum differs from {MANIFEST}; refusing to use it")
            new[fname] = digest
    if not manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text("".join(f"{d}  {f}\n" for f, d in sorted(new.items())))
        print(f"wrote {MANIFEST} (first download: checksums recorded, not independently verified)")


def load_workloads(proc: str, data_dir: Path = DATA_DIR) -> dict:
    """``{workload name: (inputs (n, f), maps (n, H, W))}``; all maps of a processor share one size."""
    out, shape = {}, None
    for fname in PROCESSORS[proc]:
        d = safe_load(data_dir / fname)
        if shape is None:
            shape = d["output"].shape[1:]
        elif d["output"].shape[1:] != shape:
            raise ValueError(f"{fname}: map size {d['output'].shape[1:]} differs from {shape}")
        out[fname.removesuffix(".pkl")] = (d["input"], d["output"])
    return out


# --------------------------------------------------------------------------- splits
def split_frames(proc: str, workloads: dict, seed: int = 0) -> dict:
    """``{"train"|"validation"|"test": [(workload, frame_index), ...]}``.

    Several workloads (i5, i7): whole workloads go to one split, chosen by a seeded shuffle, ~70/15/15
    (i7 13/3/3, i5 3/1/1). Single recording: contiguous blocks 0.68/0.14/0.14 of the frames, with a gap of
    2 frames between blocks that is used by no split, so adjacent near-duplicate frames never straddle a boundary.
    """
    names = sorted(workloads)
    if len(names) > 1:
        rng = np.random.default_rng(seed)
        order = [names[i] for i in rng.permutation(len(names))]
        n_val = n_test = max(1, round(0.15 * len(names)))
        groups = {"test": order[:n_test], "validation": order[n_test:n_test + n_val], "train": order[n_test + n_val:]}
        return {k: [(w, i) for w in sorted(v) for i in range(len(workloads[w][1]))] for k, v in groups.items()}
    name = names[0]
    n = len(workloads[name][1])
    gap = 2
    n_val = n_test = int(round(0.14 * n))
    n_train = n - n_val - n_test - 2 * gap
    a, b = n_train, n_train + gap
    c = b + n_val
    d = c + gap
    return {"train": [(name, i) for i in range(0, a)], "validation": [(name, i) for i in range(b, c)],
            "test": [(name, i) for i in range(d, n)]}


def stack(workloads: dict, frames: list) -> np.ndarray:
    return np.stack([workloads[w][1][i] for w, i in frames]).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Real thermal-map dataset helper")
    ap.add_argument("--download", action="store_true", help="download the files into data/real_thermal/")
    ap.add_argument("--processors", default=None, help="comma list of: " + ", ".join(PROCESSORS))
    ap.add_argument("--summary", action="store_true", help="print shapes and split sizes")
    args = ap.parse_args()
    procs = args.processors.split(",") if args.processors else list(PROCESSORS)
    if args.download:
        download(procs)
    if args.summary:
        for p in procs:
            wl = load_workloads(p)
            sp = split_frames(p, wl)
            shape = next(iter(wl.values()))[1].shape[1:]
            print(f"{p}: {len(wl)} recording(s), map {shape}, frames/recording "
                  f"{sorted({len(v[1]) for v in wl.values()})}, split sizes "
                  + ", ".join(f"{k}={len(v)}" for k, v in sp.items()))
            for k, v in sp.items():
                print(f"   {k}: " + (", ".join(sorted({w for w, _ in v})) if len(wl) > 1 else f"frames {v[0][1]}..{v[-1][1]}"))


if __name__ == "__main__":
    main()
