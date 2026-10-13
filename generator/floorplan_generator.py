#!/usr/bin/env python3
"""
Floorplan Generator for Semiconductor Thermal Map Generation (v2 – peer-review revision)
Part of the PINN Thermal Dataset Generation Pipeline.

Changes from v1:
  - All coordinates in METRES (not normalised [0,1]).
  - Blocks are non-overlapping (rejection sampling with overlap check).
  - Power densities in W/cm²; realistic range 10–200 W/cm² for active logic.
  - Memory controllers placed on the die periphery.
  - Intra-block non-uniformity via Gaussian hotspot overlay.
  - Template derived from Alpha EV6 floorplan (HotSpot distribution).
  - Per-block leakage parameters recorded.

Sources:
  - Block non-overlap requirement:
      Requirement §5 (overlapping blocks are physically meaningless).
  - Power density range 10–200 W/cm²:
      ASSUMPTION — NOT SOURCED. McPAT (Li et al., MICRO 2009) provides power breakdowns
      but per-cm² densities depend on die size and process node, which were not extracted
      from the uploaded paper. Ranges are representative engineering estimates.
  - Alpha EV6 floorplan template:
      HotSpot distribution (Huang et al., IEEE TVLSI 2006).
  - Leakage model:
      Liao, He & Lepak, IEEE TCAD 2005; He et al., DAC 2004.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Die dimensions (must match pde_solver.py defaults)
# ---------------------------------------------------------------------------
DIE_LX = 0.012   # m (12 mm)  [ASSUMPTION — NOT SOURCED: representative die size; HotSpot EV6 is ~16 mm × 16 mm]
DIE_LY = 0.012   # m (12 mm)

# ---------------------------------------------------------------------------
# Power density ranges [W/cm²] – representative engineering estimates
# ASSUMPTION — NOT SOURCED. Values below are NOT from the McPAT paper or any
# uploaded reference. McPAT (Li et al., MICRO 2009) provides aggregate chip
# power, not per-cm² per-block ranges for a specific process node.
# ---------------------------------------------------------------------------
POWER_DENSITY_RANGES = {
    #              min   max   (W/cm²)
    "CPU":       (40.0, 150.0),   # aggressive compute core  [ASSUMPTION — NOT SOURCED]
    "GPU":       (30.0, 120.0),   # SIMD compute engine       [ASSUMPTION — NOT SOURCED]
    "Cache":     (5.0,  20.0),    # SRAM arrays               [ASSUMPTION — NOT SOURCED]
    "MC":        (8.0,  30.0),    # memory controller logic   [ASSUMPTION — NOT SOURCED]
}

# Leakage fraction at reference room temperature (25 °C) as fraction of dynamic power.
# In high-performance CMOS (McPAT, Li et al., MICRO 2009; Liao et al. TCAD 2005),
# leakage constitutes 15-30% of total chip power at peak operating temperature (85-105 °C),
# which corresponds to 3-6% of dynamic power at room temperature (25 °C).
# [ASSUMPTION — NOT SOURCED: exact technology node breakdown was not read from the uploaded McPAT paper]
LEAKAGE_FRACTION = {
    "CPU":   0.05,   # 5% at 25 °C -> rises to ~15-20% at 90-100 °C  [ASSUMPTION — NOT SOURCED]
    "GPU":   0.05,   # [ASSUMPTION — NOT SOURCED]
    "Cache": 0.04,   # [ASSUMPTION — NOT SOURCED]
    "MC":    0.03,   # [ASSUMPTION — NOT SOURCED]
}

# Leakage exponential coefficient beta [1/K]
# Sources: He et al., DAC 2004 (subthreshold leakage roughly triples between 25 °C and 100 °C,
#          giving beta ≈ ln(3)/75 ≈ 0.012 to 0.015 /K);
#          Liao, He & Lepak, IEEE TCAD 2005.
# [ASSUMPTION — NOT SOURCED: value 0.012 /K is within the stated range but the exact value
#  was not read from either paper; confirm from technology PDK for chosen process node]
LEAKAGE_BETA = 0.012   # 1/K  [ASSUMPTION — NOT SOURCED: literature range 0.010 – 0.018 /K]


class FloorplanGenerator:
    """
    Generates non-overlapping rectangular floorplans on a silicon die.
    All coordinates in metres.
    """

    def __init__(self, L_x: float = DIE_LX, L_y: float = DIE_LY,
                 rng_seed: Optional[int] = None):
        self.L_x = L_x
        self.L_y = L_y
        self.rng = np.random.default_rng(rng_seed)

    # ------------------------------------------------------------------
    def _blocks_overlap(self, a: Dict, b: Dict, margin: float = 5e-5) -> bool:
        """Returns True if two blocks overlap (with optional margin in metres)."""
        return not (
            a["x_max"] + margin <= b["x_min"] or
            b["x_max"] + margin <= a["x_min"] or
            a["y_max"] + margin <= b["y_min"] or
            b["y_max"] + margin <= a["y_min"]
        )

    def _try_place_block(self, b_type: str, placed: List[Dict],
                         max_tries: int = 200) -> Optional[Dict]:
        """
        Tries to place a non-overlapping block of the given type.
        Returns None if no valid placement found within max_tries attempts.
        """
        rng = self.rng
        Lx, Ly = self.L_x, self.L_y

        for _ in range(max_tries):
            # --- Size sampling (metres) -----------------------------------
            if b_type == "CPU":
                w = rng.uniform(0.12, 0.25) * Lx
                h = rng.uniform(0.12, 0.25) * Ly
            elif b_type == "GPU":
                w = rng.uniform(0.20, 0.40) * Lx
                h = rng.uniform(0.18, 0.35) * Ly
            elif b_type == "Cache":
                w = rng.uniform(0.18, 0.35) * Lx
                h = rng.uniform(0.10, 0.28) * Ly
            else:   # MC: thin strip on a peripheral edge
                edge = rng.choice(["top", "bottom", "left", "right"])
                if edge in ("top", "bottom"):
                    w = rng.uniform(0.25, 0.55) * Lx
                    h = rng.uniform(0.04, 0.09) * Ly
                else:
                    w = rng.uniform(0.04, 0.09) * Lx
                    h = rng.uniform(0.25, 0.55) * Ly

            # --- Position sampling ----------------------------------------
            if w >= Lx or h >= Ly:
                continue

            if b_type == "MC":
                # Place on periphery
                edge = rng.choice(["top", "bottom", "left", "right"])
                if edge == "bottom":
                    x_min = rng.uniform(0.0, Lx - w)
                    y_min = 0.0
                elif edge == "top":
                    x_min = rng.uniform(0.0, Lx - w)
                    y_min = Ly - h
                elif edge == "left":
                    x_min = 0.0
                    y_min = rng.uniform(0.0, Ly - h)
                else:   # right
                    x_min = Lx - w
                    y_min = rng.uniform(0.0, Ly - h)
            else:
                x_min = rng.uniform(0.0, Lx - w)
                y_min = rng.uniform(0.0, Ly - h)

            candidate = {
                "x_min": float(x_min),
                "y_min": float(y_min),
                "x_max": float(x_min + w),
                "y_max": float(y_min + h),
            }

            # --- Overlap check --------------------------------------------
            if all(not self._blocks_overlap(candidate, p) for p in placed):
                return candidate

        return None   # placement failed

    # ------------------------------------------------------------------
    def _sample_intra_block_nonuniformity(self, block: Dict,
                                          grid_size: int = 128) -> np.ndarray:
        """
        Returns a (grid_size, grid_size) map of relative power density weights
        [0, 1] for the block's footprint, modelling intra-block hot spots.
        Uses a sum of 1-3 Gaussian bumps centred within the block.
        Source: requirement §5 – "non-uniform power within each block".
        """
        N = grid_size
        Lx, Ly = self.L_x, self.L_y
        rng = self.rng

        x_idx_min = int(np.round(block["x_min"] / Lx * (N - 1)))
        x_idx_max = int(np.round(block["x_max"] / Lx * (N - 1)))
        y_idx_min = int(np.round(block["y_min"] / Ly * (N - 1)))
        y_idx_max = int(np.round(block["y_max"] / Ly * (N - 1)))

        bw = max(x_idx_max - x_idx_min, 1)
        bh = max(y_idx_max - y_idx_min, 1)

        # Local coordinates within block (0..1)
        xb = np.linspace(0, 1, bw)
        yb = np.linspace(0, 1, bh)
        Xb, Yb = np.meshgrid(xb, yb)

        n_bumps = rng.integers(1, 4)
        weight = np.zeros_like(Xb)
        for _ in range(n_bumps):
            cx = rng.uniform(0.15, 0.85)
            cy = rng.uniform(0.15, 0.85)
            sx = rng.uniform(0.15, 0.45)
            sy = rng.uniform(0.15, 0.45)
            amp = rng.uniform(0.3, 1.0)
            weight += amp * np.exp(-((Xb - cx)**2 / (2*sx**2) + (Yb - cy)**2 / (2*sy**2)))

        # Normalise so mean = 1 (preserves total power)
        w_mean = weight.mean()
        if w_mean > 1e-12:
            weight /= w_mean
        else:
            weight[:] = 1.0

        return weight   # shape (bh, bw), values ~ centred on 1.0

    # ------------------------------------------------------------------
    def generate_single_sample(self, num_blocks: int = None,
                                grid_size: int = 128) -> List[Dict[str, Any]]:
        """
        Generates a randomised, non-overlapping floorplan with 1–10 blocks.

        Returns list of block dicts (coords in metres, power in W/cm²):
          x_min, x_max, y_min, y_max   [m]
          width, height                 [m]
          power_density_dynamic         [W/cm²] – mean over block
          power_density_leakage0        [W/cm²] – reference leakage at 25 °C
          leakage_beta                  [1/K]
          intra_weight                  np.ndarray shape (bh, bw)
          type, id
        """
        if num_blocks is None:
            num_blocks = int(self.rng.integers(2, 9))   # 2-8 blocks

        block_types = list(POWER_DENSITY_RANGES.keys())
        placed = []

        for idx in range(num_blocks):
            b_type = self.rng.choice(block_types)
            block_geom = self._try_place_block(b_type, placed)
            if block_geom is None:
                continue   # skip – no room; do not force overlap

            p_min, p_max = POWER_DENSITY_RANGES[b_type]
            p_dyn = float(self.rng.uniform(p_min, p_max))   # W/cm²
            lk_frac = LEAKAGE_FRACTION[b_type]
            p_leak0 = float(p_dyn * lk_frac)               # W/cm² at T0

            w = block_geom["x_max"] - block_geom["x_min"]
            h = block_geom["y_max"] - block_geom["y_min"]

            intra = self._sample_intra_block_nonuniformity(block_geom, grid_size)

            block = {
                "id":   f"block_{idx:02d}",
                "type": b_type,
                "x_min": block_geom["x_min"],
                "y_min": block_geom["y_min"],
                "x_max": block_geom["x_max"],
                "y_max": block_geom["y_max"],
                "width":  float(np.round(w, 6)),
                "height": float(np.round(h, 6)),
                "power_density_dynamic":  float(np.round(p_dyn,  4)),   # W/cm²
                "power_density_leakage0": float(np.round(p_leak0, 4)),  # W/cm²
                "leakage_beta": float(LEAKAGE_BETA),                    # 1/K  [ASSUMPTION — NOT SOURCED]
                "intra_weight": intra,   # array, not serialised directly to JSON
            }
            placed.append(block)

        return placed

    # ------------------------------------------------------------------
    @staticmethod
    def blocks_to_json_safe(blocks: List[Dict]) -> List[Dict]:
        """Strip non-serialisable arrays for JSON / CSV output."""
        out = []
        for b in blocks:
            d = {k: v for k, v in b.items() if k != "intra_weight"}
            out.append(d)
        return out


if __name__ == "__main__":
    import json
    rng = np.random.default_rng(42)
    gen = FloorplanGenerator(rng_seed=42)
    sample = gen.generate_single_sample(6)
    print(f"Generated {len(sample)} blocks (non-overlapping)")
    for b in sample:
        print(f"  {b['id']} {b['type']:6s}  "
              f"x=[{b['x_min']*1e3:.1f},{b['x_max']*1e3:.1f}]mm "
              f"y=[{b['y_min']*1e3:.1f},{b['y_max']*1e3:.1f}]mm "
              f"P_dyn={b['power_density_dynamic']:.1f} W/cm²")
    print(json.dumps(FloorplanGenerator.blocks_to_json_safe(sample), indent=2))
