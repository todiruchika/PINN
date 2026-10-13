#!/usr/bin/env python3
"""
Realistic Sparse Sensor Generator (v2 – peer-review revision)
Part of the PINN Thermal Dataset Generation Pipeline.

Changes from v1:
  - Sensor positions are chosen at block centres or near expected hotspots,
    NOT uniformly random.  Source: Intel Core Duo on-die sensors
    (arXiv 0709.1861, §3.1): "sensors are placed at design time near
    expected hot spots but do not always capture the hottest point."
  - Sensors have explicit Gaussian noise and A/D quantization.
    Noise model:
      sigma_analogue ≈ 0.5 °C  (A/D quantization, Intel Core Duo arXiv 0709.1861)
      sigma_digital  ≈ 5.0 °C  (digital thermal sensors, Intel Agilex 7 AN 944)
    We use sigma = 1.0 °C as a conservative mid-range default, with
    1 °C quantization step.  [TBD: choose sigma from a specific sensor spec.]
  - Calibration offset sampled per sensor from Uniform(−2, +2) °C.
  - Sensors kept ≥ 3 grid cells from the boundary.
  - No duplicate sensors.json default file (three densities only: 5, 8, 10).

Sources:
  arXiv 0709.1861 – Intel Core Duo on-die sensor characterisation.
  Intel Agilex 7 Thermal Design Guide AN 944 – digital thermal sensor accuracy.
  Requirement §6 (noise model, fixed positions, quantization).
"""

import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Noise model parameters
# ---------------------------------------------------------------------------
# Analogue sensor noise (A/D quantization, Intel Core Duo):
# arXiv 0709.1861: ±0.5 °C resolution floor from A/D.
SIGMA_ANALOGUE_C = 0.5   # °C (1-sigma Gaussian noise, Intel Core Duo)

# Digital thermal sensor accuracy (Intel Agilex 7 AN 944): ≈ ±5 °C.
SIGMA_DIGITAL_C  = 5.0   # °C [TBD: confirm exact spec from AN 944 datasheet]

# Default noise model: we use sigma = 1.0 °C (slightly conservative analogue).
# [TBD: update once sensor spec for the target platform is confirmed.]
SIGMA_DEFAULT_C  = 1.0   # °C

# Quantization step (1 °C matches Intel Core Duo A/D resolution)
QUANT_STEP_C = 1.0   # °C (arXiv 0709.1861)

# Calibration offset range per sensor
CALIB_OFFSET_RANGE_C = 2.0   # ±2 °C uniformly drawn per sensor

# Minimum grid cells from boundary
SENSOR_BOUNDARY_MARGIN = 3   # cells


class SensorGenerator:
    """
    Places sensors near block centres / hotspots, adds Gaussian noise and
    quantization, records calibration offsets.

    Sensor placement strategy:
      1. For each block, place a sensor near its centre (within ±10% of
         block half-width, clamped to grid bounds and boundary margin).
      2. If more sensors are requested than blocks, fill remaining slots
         with randomly chosen interior points (avoiding previous positions).
      3. If fewer sensors than blocks, subsample the block list randomly.
    """

    def __init__(self, grid_size: int = 128,
                 L_x: float = 0.012, L_y: float = 0.012,
                 sigma_C: float = SIGMA_DEFAULT_C,
                 quant_step_C: float = QUANT_STEP_C,
                 rng_seed: Optional[int] = None):
        self.N = grid_size
        self.L_x = L_x
        self.L_y = L_y
        self.sigma_C = sigma_C
        self.quant_step_C = quant_step_C
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    def _block_centre_grid_idx(self, block: Dict) -> tuple:
        """Returns (row, col) grid index for a block's geometric centre."""
        N = self.N
        cx = (block["x_min"] + block["x_max"]) / 2
        cy = (block["y_min"] + block["y_max"]) / 2
        col = int(np.round(cx / self.L_x * (N - 1)))
        row = int(np.round(cy / self.L_y * (N - 1)))
        col = np.clip(col, SENSOR_BOUNDARY_MARGIN, N - 1 - SENSOR_BOUNDARY_MARGIN)
        row = np.clip(row, SENSOR_BOUNDARY_MARGIN, N - 1 - SENSOR_BOUNDARY_MARGIN)
        return int(row), int(col)

    def _jitter_position(self, row: int, col: int, jitter_cells: int = 3) -> tuple:
        """Adds small random jitter to the sensor location."""
        N = self.N
        dr = int(self.rng.integers(-jitter_cells, jitter_cells + 1))
        dc = int(self.rng.integers(-jitter_cells, jitter_cells + 1))
        row = np.clip(row + dr, SENSOR_BOUNDARY_MARGIN, N - 1 - SENSOR_BOUNDARY_MARGIN)
        col = np.clip(col + dc, SENSOR_BOUNDARY_MARGIN, N - 1 - SENSOR_BOUNDARY_MARGIN)
        return int(row), int(col)

    # ------------------------------------------------------------------
    def generate_sensors_for_count(self, temperature_matrix: np.ndarray,
                                   blocks: List[Dict], count: int) -> Dict[str, Any]:
        """
        Places 'count' sensors, preferentially near block centres.
        Returns dict with:
          sensor_count, positions [[x_phys, y_phys] …],
          temperatures_true (exact grid values),
          temperatures_noisy (noisy observed readings),
          noise_sigma_C, quant_step_C,
          calibration_offsets (per-sensor systematic error in °C).
        """
        N = self.N
        placed = set()
        rows, cols = [], []

        # 1. Block-centre positions (primary placement)
        block_indices = list(range(len(blocks)))
        self.rng.shuffle(block_indices)
        for bi in block_indices:
            if len(rows) >= count:
                break
            r, c = self._block_centre_grid_idx(blocks[bi])
            r, c = self._jitter_position(r, c)
            key = (r, c)
            if key not in placed:
                placed.add(key)
                rows.append(r)
                cols.append(c)

        # 2. Fill remaining with random interior points
        max_tries = 5000
        tries = 0
        while len(rows) < count and tries < max_tries:
            r = int(self.rng.integers(SENSOR_BOUNDARY_MARGIN, N - SENSOR_BOUNDARY_MARGIN))
            c = int(self.rng.integers(SENSOR_BOUNDARY_MARGIN, N - SENSOR_BOUNDARY_MARGIN))
            key = (r, c)
            if key not in placed:
                placed.add(key)
                rows.append(r)
                cols.append(c)
            tries += 1

        # Build output arrays
        positions       = []
        temps_true      = []
        temps_noisy     = []
        calib_offsets   = []

        for r, c in zip(rows, cols):
            x_phys = float(c / (N - 1) * self.L_x)
            y_phys = float(r / (N - 1) * self.L_y)
            positions.append([x_phys, y_phys])

            T_true = float(temperature_matrix[r, c])
            temps_true.append(T_true)

            # Calibration offset per sensor (systematic bias)
            calib = float(self.rng.uniform(-CALIB_OFFSET_RANGE_C, CALIB_OFFSET_RANGE_C))
            calib_offsets.append(round(calib, 3))

            # Gaussian noise + quantization
            noise = float(self.rng.normal(0.0, self.sigma_C))
            T_obs = T_true + calib + noise
            # Quantize to nearest quant_step_C
            T_obs = round(T_obs / self.quant_step_C) * self.quant_step_C
            temps_noisy.append(float(T_obs))

        return {
            "sensor_count":       count,
            "positions":          positions,        # [[x_m, y_m], …]
            "temperatures_true":  [round(t, 4) for t in temps_true],
            "temperatures_noisy": [round(t, 3) for t in temps_noisy],
            "noise_sigma_C":      self.sigma_C,
            "quant_step_C":       self.quant_step_C,
            "calibration_offsets_C": calib_offsets,
            "placement_strategy": "block_centre_with_jitter",
            "boundary_margin_cells": SENSOR_BOUNDARY_MARGIN,
            # Citations
            "_source_noise": "arXiv 0709.1861 (Intel Core Duo A/D quantization)",
            "_source_digital_sensor": "Intel Agilex 7 AN 944 [TBD: confirm accuracy spec]",
        }

    # ------------------------------------------------------------------
    def process_sample(self, temperature_matrix: np.ndarray,
                       blocks: List[Dict]) -> Dict[str, Any]:
        """
        Generates sensor readouts for all three required densities: 5, 8, 10.
        Returns a dict with keys sensors_5, sensors_8, sensors_10.
        No duplicate default sensors.json is written (caller decides).
        """
        return {
            "sensors_5":  self.generate_sensors_for_count(temperature_matrix, blocks, 5),
            "sensors_8":  self.generate_sensors_for_count(temperature_matrix, blocks, 8),
            "sensors_10": self.generate_sensors_for_count(temperature_matrix, blocks, 10),
        }


if __name__ == "__main__":
    import json
    N = 128
    dummy_T = np.ones((N, N)) * 45.0
    dummy_T[40:70, 40:70] = 85.0  # hotspot
    dummy_T[80:100, 20:50] = 70.0

    dummy_blocks = [
        {"x_min": 0.004, "x_max": 0.008, "y_min": 0.004, "y_max": 0.007},
        {"x_min": 0.009, "x_max": 0.011, "y_min": 0.002, "y_max": 0.006},
    ]

    gen = SensorGenerator(grid_size=N, rng_seed=0)
    result = gen.process_sample(dummy_T, dummy_blocks)
    for key in ["sensors_5", "sensors_8", "sensors_10"]:
        s = result[key]
        print(f"{key}: {s['sensor_count']} sensors, "
              f"T_noisy range: [{min(s['temperatures_noisy']):.1f}, "
              f"{max(s['temperatures_noisy']):.1f}] °C")
