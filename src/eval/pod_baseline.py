"""Plain POD estimate from the sensor readings, scored on the test chips.

Fits the mean and POD modes on the TRAINING chips (rise above ambient, no per-chip
normalisation), then fits the modes to each test chip's readings by least squares.
This is the starting point the FNO/PINO residual model uses (--prior pod).

Usage (from the repo root):
    python -m src.eval.pod_baseline
Note: src/eval/gappy_pod.py uses a basis fitted to normalised temperatures and is not
used here.
"""
import json, numpy as np, pandas as pd
from src.data.splits import load_splits
from src.data.dataset import load_metadata, load_sample, GENERATED_SENSOR_DIR
from src.eval.metrics import all_metrics
from src.sensors.readers import sensor_mask, metre_to_idx
from src.data.dataset import RAW_DIR
meta = load_metadata()
ambient = dict(zip(meta.sample_id, meta.T_amb_C))
splits = load_splits()
# Basis from TRAINING chips only: rise above ambient, no per-chip normalisation.
X = np.stack([np.load(RAW_DIR / "train/matrices" / f"{s}.npy").ravel() - ambient[s] for s in splits["train"]], axis=1)
mean = X.mean(axis=1, keepdims=True)
U, S, _ = np.linalg.svd(X - mean, full_matrices=False)
print("energy in first 5/10/20 modes:", [round(float((S[:k]**2).sum()/(S**2).sum()),3) for k in (5,10,20)])
for r in (10, 20):
    basis = U[:, :r]
    for n in (5, 8, 10):
        rows = []
        for sid in splits["test"]:
            s = load_sample(sid, n, meta=meta)
            pos = s["sensors"]["positions"]; vals = np.array(s["sensors"]["temperatures_noisy"]) - ambient[sid]
            idx = np.array([metre_to_idx(y)*128 + metre_to_idx(x) for x, y in pos])
            coeffs = np.linalg.lstsq(basis[idx], vals - mean[idx, 0], rcond=1e-3)[0]
            pred = (mean[:, 0] + basis @ coeffs).reshape(128, 128) + ambient[sid]
            mask = ~sensor_mask(pos)
            rows.append(all_metrics(pred, s["T"], mask))
        m = {k: np.mean([x[k] for x in rows]) for k in rows[0]}
        print(f"r={r} N={n}: relL2={m['relative_l2']*100:.2f}%  MAE={m['mae_C']:.2f}  peak={m['peak_temp_error_C']:.2f}  hotspot={m['hotspot_localization_cells']:.1f}", flush=True)
