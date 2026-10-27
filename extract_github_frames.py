"""Step 1: turn the kept GitHub thermal-map .pkl files into the directory layout
pure_pinn/src/data_loader.py expects (matrices/*.npy + metadata.csv), so the
existing PINN pipeline needs minimal changes to run on real-chip data.

Frames are kept at their NATIVE resolution per device (no cropping, no
resizing) -- data_loader.py's grid helpers and evaluate.py's metrics have been
generalised to accept an (H, W) shape instead of assuming a square n x n grid,
so every real measured pixel is preserved.

Approximations made here (see conversation notes / writeup limitations):
  * T_amb is estimated per-frame as the 5th percentile temperature (coolest
    region), since no ambient sensor reading is provided.
  * die_L_x_m is set to 1.0 (abstract unit square) -- no physical die size is
    known for these parts; coordinates stay as fractions of the image.
  * block_count = 0, blocks_json = "[]" -- no floorplan/power ground truth
    exists for real chips, so source-level metrics degrade to NaN downstream.
  * Frames 0-69 of each file -> "train" split (POD-QR basis fit only).
    Frames 70-99            -> "test" split  (sensors read + scored).

Run from the repository root:
    python extract_github_frames.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "github_dataset" / "commercial_thermal_map_dataset" / "data_files"
OUT_DIR = ROOT / "data" / "github_real"

# device_tag -> pkl filename. One basis will be fit per device_tag.
FILES = {
    "i5_flac": "CPU_i5_flac.pkl",
    "i5_7zip": "CPU_i5_7zip.pkl",
    "i5_aobench": "CPU_i5_aobench.pkl",
    "i5_opencv": "CPU_i5_opencv.pkl",
    "i5_stream": "CPU_i5_stream.pkl",
    "i7_idle": "CPU_i7_idle.pkl",
    "i7_aobench": "CPU_i7_aobench.pkl",
    "i7_compress7zip": "CPU_i7_compress-7zip.pkl",
    "tpu_edge": "TPU_Google_Edge.pkl",
}

N_TRAIN = 70  # frames 0..69 -> train (POD-QR fit only)


def main() -> None:
    rows = {"train": [], "test": []}

    for device_tag, fname in FILES.items():
        path = SRC_DIR / fname
        if not path.exists():
            print(f"skip {device_tag}: {path} not found")
            continue

        with open(path, "rb") as f:
            d = pickle.load(f)
        out = np.asarray(d["output"])
        n_frames = out.shape[0]
        print(f"{device_tag}: {n_frames} frames, native shape {out.shape[1:]}")

        for i in range(n_frames):
            split = "train" if i < N_TRAIN else "test"
            frame = out[i].astype(np.float64)  # native (H, W), no crop/resize
            sample_id = f"{device_tag}_{i:03d}"

            mat_dir = OUT_DIR / split / "matrices"
            mat_dir.mkdir(parents=True, exist_ok=True)
            np.save(mat_dir / f"{sample_id}.npy", frame.astype(np.float32))

            t_amb = float(np.percentile(frame, 5))
            rows[split].append({
                "sample_id": sample_id,
                "device": device_tag,
                "frame_index": i,
                "T_amb_C": t_amb,
                "die_L_x_m": 1.0,
                "peak_temp_C": float(frame.max()),
                "min_temp_C": float(frame.min()),
                "block_count": 0,
                "blocks_json": "[]",
                "grid_h": int(frame.shape[0]),
                "grid_w": int(frame.shape[1]),
            })

    for split in ("train", "test"):
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows[split])
        df.to_csv(split_dir / "metadata.csv", index=False)
        print(f"{split}: {len(df)} samples -> {split_dir / 'metadata.csv'}")
        if len(df):
            print(df.groupby("device")[["grid_h", "grid_w"]].first())


if __name__ == "__main__":
    main()
