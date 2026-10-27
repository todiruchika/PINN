"""Step 2: fit a POD-QR sensor layout per device from its 70 "train" frames
(data/github_real/train), then write sensor-reading JSON files for its 30 held-
out "test" frames (data/github_real/test), in the exact schema
pure_pinn/src/data_loader.py's ChipSample.sensors() expects.

One layout per device: resolutions differ across devices (i5: 177x166,
i7: 185x154, TPU: 256x244), so snapshots cannot be pooled into one SVD. Within
a device every frame shares the same (H, W), so POD-QR applies directly, same
as src/sensors/podqr.py -- just parameterised over device/grid-shape instead
of hardcoded to the synthetic 128x128 case.

The QR pivot order is reused directly for N=5/8/10 (first 5 pivots are a
subset of the first 8, etc.), same nesting property as the synthetic layout.

No synthetic noise is added to the sensor readings here: real IR-camera noise
is already present in the measurement itself, unlike the synthetic generator
which has to inject noise deliberately.

Run from the repository root (after extract_github_frames.py):
    python generate_github_sensors.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr

ROOT = Path(__file__).parent
REAL_ROOT = ROOT / "data" / "github_real"
SENSOR_ROOT = ROOT / "data" / "generated_github"
LAYOUT_DIR = SENSOR_ROOT / "layouts"

ENERGY_THRESHOLD = 0.95
MIN_R = 10          # keep at least enough modes for the N=10 sensor set
MARGIN = 3          # keep sensors this many pixels off every edge
SENSOR_COUNTS = (5, 8, 10)


def fit_device_layout(device: str, train_df: pd.DataFrame) -> dict:
    """POD-QR layout for one device, fit from its train-split frames only."""
    H, W = int(train_df["grid_h"].iloc[0]), int(train_df["grid_w"].iloc[0])

    snapshots = []
    for _, row in train_df.iterrows():
        frame = np.load(REAL_ROOT / "train" / "matrices" / f"{row['sample_id']}.npy").astype(np.float64)
        t_amb = float(row["T_amb_C"])
        denom = max(frame.max() - t_amb, 1e-6)
        snapshots.append(((frame - t_amb) / denom).ravel())

    X = np.stack(snapshots, axis=1)          # (H*W, n_train)
    mean = X.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(X - mean, full_matrices=False)
    cumulative = np.cumsum(s ** 2) / (s ** 2).sum()
    r = max(int(np.searchsorted(cumulative, ENERGY_THRESHOLD)) + 1, MIN_R)
    r = min(r, U.shape[1])
    Psi = U[:, :r]

    mask = np.zeros((H, W), dtype=bool)
    mask[MARGIN:H - MARGIN, MARGIN:W - MARGIN] = True
    valid_flat = np.where(mask.ravel())[0]

    Psi_valid = Psi[valid_flat, :]
    _, _, pivot_in_valid = qr(Psi_valid.T, pivoting=True, mode="economic")
    sensor_flat = valid_flat[pivot_in_valid[:max(SENSOR_COUNTS)]]
    rows, cols = np.unravel_index(sensor_flat, (H, W))

    layout = {
        "device": device, "grid_h": H, "grid_w": W, "r": r,
        "cumulative_energy_at_r": float(cumulative[min(r, len(cumulative)) - 1]),
        "n_train_frames": len(train_df),
        "pivots": [{"row": int(rows[i]), "col": int(cols[i])} for i in range(len(rows))],
    }
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LAYOUT_DIR / f"{device}_layout.json", "w") as f:
        json.dump(layout, f, indent=2)
    print(f"{device}: r={r} (energy={layout['cumulative_energy_at_r']:.4f}), "
          f"pivots -> {LAYOUT_DIR / f'{device}_layout.json'}")
    return layout


def write_sensor_files(device: str, layout: dict, test_df: pd.DataFrame) -> int:
    H, W = layout["grid_h"], layout["grid_w"]
    pivots = layout["pivots"]
    n_written = 0

    for _, row in test_df.iterrows():
        sample_id = row["sample_id"]
        frame = np.load(REAL_ROOT / "test" / "matrices" / f"{sample_id}.npy").astype(np.float64)

        for n_sensors in SENSOR_COUNTS:
            chosen = pivots[:n_sensors]
            positions_m = [[c["col"] / (W - 1), c["row"] / (H - 1)] for c in chosen]  # die_L=1.0
            temps = [float(frame[c["row"], c["col"]]) for c in chosen]

            out_dir = SENSOR_ROOT / "test"
            out_dir.mkdir(parents=True, exist_ok=True)
            payload = {"sample_id": sample_id, "n_sensors": n_sensors,
                       "positions": positions_m, "temperatures_noisy": temps}
            with open(out_dir / f"{sample_id}_sensors_{n_sensors}.json", "w") as f:
                json.dump(payload, f, indent=2)
            n_written += 1
    return n_written


def main() -> None:
    train_meta = pd.read_csv(REAL_ROOT / "train" / "metadata.csv")
    test_meta = pd.read_csv(REAL_ROOT / "test" / "metadata.csv")

    total = 0
    for device, train_df in train_meta.groupby("device"):
        layout = fit_device_layout(device, train_df)
        test_df = test_meta[test_meta["device"] == device]
        n = write_sensor_files(device, layout, test_df)
        total += n
        print(f"  wrote {n} sensor files ({len(test_df)} frames x {len(SENSOR_COUNTS)} sensor counts)")

    print(f"\ntotal sensor files written: {total} -> {SENSOR_ROOT / 'test'}")


if __name__ == "__main__":
    main()
