| Model | Sensors | Seeds | Steps | Batch | Chips seen | Relative L2 (%) | MAE (°C) | Peak error (°C) | Hotspot error (cells) | Params |
|---|---|---|---|---|---|---|---|---|---|---|
| FNO/PINO (final) | 5 | 1 | 3000 | 8 | 24000 | 13.20 ± 0.00 | 5.46 | -14.02 | 34.5 | 533186 |
| FNO/PINO (final) | 8 | 1 | 3000 | 8 | 24000 | 10.79 ± 0.00 | 4.45 | -9.76 | 28.1 | 533186 |
| FNO/PINO (final) | 10 | 3 | 3000 | 8 | 24000 | 9.78 ± 0.20 | 4.02 | -8.48 | 20.6 | 533186 |
| FNO/PINO + POD estimate | 5 | 1 | 3000 | 8 | 24000 | 11.23 ± 0.00 | 4.50 | -9.80 | 33.5 | 533218 |
| FBPINN, 3000 steps x 8 chips (full grid) | 5 | 1 | 3000 | 8 | 24000 | 12.25 ± 0.00 | 5.12 | -15.00 | 48.3 | 23346 |
| FBPINN, 3000 steps x 8 chips (full grid) | 8 | 1 | 3000 | 8 | 24000 | 10.02 ± 0.00 | 4.17 | -12.58 | 39.9 | 23922 |
| FBPINN, 3000 steps x 8 chips (full grid) | 10 | 1 | 3000 | 8 | 24000 | 8.11 ± 0.00 | 3.35 | -10.52 | 29.5 | 24306 |
| FBPINN fast (batch 32, 1024 points; 5 sensors: 10k steps, 8 and 10: 6k) | 5 | 1 | 10000 | 32 | 320000 | 11.35 ± 0.00 | 4.64 | -13.20 | 30.7 | 23346 |
| FBPINN fast (batch 32, 1024 points; 5 sensors: 10k steps, 8 and 10: 6k) | 8 | 3 | 6000 | 32 | 192000 | 8.95 ± 0.05 | 3.58 | -10.62 | 27.3 | 23922 |
| FBPINN fast (batch 32, 1024 points; 5 sensors: 10k steps, 8 and 10: 6k) | 10 | 3 | 6000 | 32 | 192000 | 6.76 ± 0.03 | 2.74 | -8.80 | 18.2 | 24306 |
| FBPINN fast + POD estimate (10k steps) | 5 | 1 | 10000 | 32 | 320000 | 11.49 ± 0.00 | 4.67 | -10.15 | 34.4 | 23346 |
| FBPINN fast, encoder 256 (10 sensors) | 10 | 1 | 6000 | 32 | 192000 | 6.80 ± 0.00 | 2.75 | -8.02 | 21.2 | 119826 |
| FBPINN fast, 16 windows (10 sensors) | 10 | 1 | 6000 | 32 | 192000 | 6.25 ± 0.00 | 2.47 | -7.55 | 18.4 | 40000 |
| FBPINN fast, physics off (10 sensors) | 10 | 1 | 6000 | 32 | 192000 | 6.83 ± 0.00 | 2.77 | -7.31 | 19.7 | 24306 |
| XPINN fast, physics on, unmasked (first version) | 5 | 1 | 10000 | 32 | 320000 | 16.62 ± 0.00 | 7.49 | -25.02 | 49.9 | 23346 |
| XPINN fast, physics on, unmasked (first version) | 10 | 1 | 6000 | 32 | 192000 | 16.17 ± 0.00 | 7.35 | -24.85 | 53.0 | 24306 |
| XPINN fast, physics on, unmasked + POD estimate | 5 | 1 | 10000 | 32 | 320000 | 17.21 ± 0.00 | 7.46 | -7.55 | 41.3 | 23346 |
| XPINN fast, 96 hidden units | 10 | 1 | 6000 | 32 | 192000 | 16.11 ± 0.00 | 7.31 | -24.84 | 53.4 | 149650 |
| XPINN fast, physics off | 5 | 1 | 6000 | 32 | 192000 | 11.38 ± 0.00 | 4.62 | -12.17 | 33.6 | 23346 |
| XPINN fast, physics off | 8 | 1 | 6000 | 32 | 192000 | 8.63 ± 0.00 | 3.39 | -8.46 | 26.6 | 23922 |
| XPINN fast, physics off | 10 | 1 | 6000 | 32 | 192000 | 6.40 ± 0.00 | 2.54 | -6.98 | 20.1 | 24306 |
| XPINN fast, physics on, tile-edge nodes masked | 5 | 1 | 6000 | 32 | 192000 | 11.43 ± 0.00 | 4.66 | -13.06 | 34.5 | 23346 |
| XPINN fast, physics on, tile-edge nodes masked | 8 | 1 | 6000 | 32 | 192000 | 8.72 ± 0.00 | 3.42 | -9.31 | 27.7 | 23922 |
| XPINN fast, physics on, tile-edge nodes masked | 10 | 3 | 6000 | 32 | 192000 | 6.45 ± 0.03 | 2.57 | -8.31 | 19.7 | 24306 |
| XPINN fast, patch-averaged physics, tile-edge nodes masked | 10 | 1 | 6000 | 32 | 192000 | 6.37 ± 0.00 | 2.52 | -7.73 | 19.9 | 24306 |
| DeepONet (teammates, reference) | 5 | 5 | 6400 | 32 | 200000 | 11.46 |  |  |  | 118017 |
| DeepONet (teammates, reference) | 8 | 5 | 6400 | 32 | 200000 | 8.62 |  |  |  | 118017 |
| DeepONet (teammates, reference) | 10 | 5 | 6400 | 32 | 200000 | 6.13 |  |  |  | 118017 |
