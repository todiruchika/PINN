# Real IR-camera data: reconstruction from virtual sensors

Branch `siya-real-data`. Data: Lu and Tan, "Thermal Map Dataset for Commercial Multi/Many Core CPU/GPU/TPU" (MLCAD 2024),
public sample at https://github.com/sheldonucr/commercial_thermal_map_dataset. Code: `src/data/real_thermal.py`,
`src/train/real_runner.py`, `src/eval/real_baselines.py`, `src/eval/real_report.py`.

## What was done

- **Chip and frames:** Intel i7-8650U, 19 workloads x 100 consecutive frames (185 x 154 IR maps, 33 to 97 degC).
  The performance-counter inputs are not used.
- **Split by workload:** 13 train, 3 validation (`aio-stress`, `phpbench`, `compress-7zip`), 3 test (`fio`, `encode-flac`, `t-test1`).
  Workloads were ranked by mean temperature and every 6th went to test or validation, so each split spans cold to hot.
  Test heat patterns are never seen in training (1300 / 300 / 300 frames).
- **Virtual sensors:** POD-QR pivots fitted on the training frames only (nested 5 < 8 < 10), read from the map, plus 1 degC Gaussian noise.
- **Models (data only, no physics):** FNO/PINO (3000 steps, batch 8), FBPINN and XPINN (6000 steps, batch 32, 1024 points),
  and the data-only decoder (6000 steps, batch 32, sensor noise 1 degC, dropout 0.1, weight decay 0.05).
  Physics and source weights are 0 (the runner refuses otherwise): a real chip has no known heat source or material constants.
- **Baselines (no neural network):** mean training map, POD fit of the sensor readings, thin-plate-spline interpolation.
- Metrics are the project metrics on non-sensor nodes, as for every other model.

## Results (test set, one seed each)

| Method | 5 sensors | 8 sensors | 10 sensors |
|---|---|---|---|
| Data-only decoder | **3.31%** | **2.75%** | 2.57% |
| FBPINN | 3.35% | 2.89% | 2.55% |
| XPINN | 3.73% | 3.44% | 3.47% |
| FNO/PINO | 4.26% | 3.87% | 4.24% |
| Thin-plate spline | 7.71% | 3.23% | 2.41% |
| POD fit | 26.05% | 19.36% | **1.96%** |
| Mean training map | 31.31% | 31.31% | 31.31% |

Relative L2 on absolute temperature in degC (the project convention), so values are small because the maps sit at 35 to 95 degC.
MAE is in `summary.md` (about 1.1 to 1.6 degC for the best networks). Figure: `error_vs_sensors.png`. Heatmaps (truth, models, spline, error maps): `heatmaps_N5.png`, `heatmaps_N8.png`, `heatmaps_N10.png`.

## What the results say

- With 5 and 8 sensors, the data-only decoder and FBPINN clearly beat the spline (3.3% vs 7.7% at 5 sensors).
- With 10 sensors the training-free baselines are as good or better: spline 2.41%, POD fit 1.96%, best network 2.55%.
  The maps are close to low-dimensional, so 10 readings nearly pin down a POD fit. A network adds little there.
- The data-only decoder and FBPINN are within about 0.1 points at 5 and 10 sensors, which is likely inside the seed spread (one seed each).
- XPINN and FNO/PINO are worse here than on the synthetic chips.
- Hotspot position is often wrong: 13 to 47 grid cells of localisation error for the networks (`summary.md`).
- On the cold, near-idle `fio` workload the sensor readings barely change, and the models put the warm region in the wrong place (see `heatmaps_N5.png`).
- The data-only decoder shows grid-like artifacts at 5 sensors.

## This does not test the physics claim

No heat-equation term was used, because the equation needs a heat source and material constants that real chips do not provide here.
So these are data-driven models against each other and against interpolation. They say nothing about whether physics helps.

## Limitations

- One seed per model. The 0.1 to 0.5 point differences between the top models are not established.
- One chip (i7-8650U). The i5, 4800U, RTX 4060 and TPU files are not used. The 4800U and RTX 4060 frames show circuit and label patterns, not heat.
- Frames inside a workload are strongly correlated (consecutive IR frames), so 1300 training frames carry far less information than 1300 independent chips.
- Assumptions that are not from the data: ambient temperature fixed at 25 degC (not recorded); frames squashed to 128 x 128, which distorts the aspect ratio;
  1 degC sensor noise; virtual sensors read from the camera map, which are not real on-die sensor readings.
- Only 3 test workloads, so results depend on which workloads were held out.

## Reproduce

```
# data: copy data_files/CPU_i7_*.pkl from the dataset repo into data/real_raw/ (first run reads and caches them)
python -m src.eval.real_baselines
python -m src.train.real_runner fno    --sensors 10 --steps 3000 --batch 8 --w-phys 0 --w-src 0 --out results/real/fno
python -m src.train.real_runner fbpinn --sensors 10 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --w-phys 0 --w-src 0 --out results/real/fbpinn
python -m src.train.real_runner xpinn  --sensors 10 --steps 6000 --batch 32 --points 1024 --phys-chips 2 --w-phys 0 --w-src 0 --out results/real/xpinn
python -m src.train.real_runner hybrid --mode ml --sensors 10 --steps 6000 --w-src 0 --noise-c 1.0 --dropout 0.1 --wd 0.05 --out results/real/decoder
python -m src.eval.real_report
```
