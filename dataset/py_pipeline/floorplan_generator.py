#!/usr/bin/env python3
"""
Floorplan Generator for Semiconductor Thermal Map Generation
Part of the PINN Thermal Dataset Generation Pipeline.

This module generates realistic floorplan architectures for a 2D semiconductor
chip of domain [0,1] x [0,1]. It produces randomized placements of active logic
components (CPU, GPU, Cache, Memory Controllers) with variable power densities.
"""

import random
import numpy as np
import json
from typing import List, Dict, Any

class FloorplanGenerator:
    def __init__(self, domain_size: float = 1.0):
        self.domain_size = domain_size
        # Realistic relative power densities in W/m^2 (scaled for temperature output of 50-110 C)
        # Assuming thermal conductivity k ~ 120.0
        self.power_densities = {
            "CPU": {"min": 1.2e7, "max": 2.5e7},  # High intensity hotspots
            "GPU": {"min": 1.0e7, "max": 1.8e7},  # Wide high-intensity areas
            "Cache": {"min": 1.0e6, "max": 3.0e6}, # Low power density
            "MC": {"min": 4.0e6, "max": 8.0e6}     # Medium density IO blocks
        }

    def generate_single_sample(self, num_blocks: int = None) -> List[Dict[str, Any]]:
        """
        Generates a randomized floorplan containing 1 to 10 functional blocks on the chip.
        """
        if num_blocks is None:
            num_blocks = random.randint(1, 10)

        blocks = []
        
        # We can place specific core blocks first to guarantee structural realism
        block_types = ["CPU", "GPU", "Cache", "MC"]
        
        for idx in range(num_blocks):
            # Select component type
            b_type = random.choice(block_types)
            p_range = self.power_densities[b_type]
            power = random.uniform(p_range["min"], p_range["max"])
            
            # Formulate coordinates based on component type typical shapes
            if b_type == "CPU":
                # CPU cores are typically square-ish, medium size
                w = random.uniform(0.12, 0.25)
                h = random.uniform(0.12, 0.25)
            elif b_type == "GPU":
                # GPU cores are typically larger rectangles
                w = random.uniform(0.25, 0.45)
                h = random.uniform(0.20, 0.40)
            elif b_type == "Cache":
                # Cache blocks can be bulky arrays matching CPU clusters
                w = random.uniform(0.20, 0.35)
                h = random.uniform(0.15, 0.30)
            else: # Memory controller
                # Thin strips along the edge of the chip
                edge = random.choice(["top", "bottom", "left", "right"])
                if edge in ["top", "bottom"]:
                    w = random.uniform(0.30, 0.60)
                    h = random.uniform(0.04, 0.10)
                else:
                    w = random.uniform(0.04, 0.10)
                    h = random.uniform(0.30, 0.60)
            
            # Randomize anchor within bounds [0, 1.0 - width/height]
            x_min = random.uniform(0.0, self.domain_size - w)
            y_min = random.uniform(0.0, self.domain_size - h)
            x_max = x_min + w
            y_max = y_min + h
            
            blocks.append({
                "id": f"block_{idx:02d}",
                "type": b_type,
                "x_min": float(np.round(x_min, 4)),
                "y_min": float(np.round(y_min, 4)),
                "x_max": float(np.round(x_max, 4)),
                "y_max": float(np.round(y_max, 4)),
                "width": float(np.round(w, 4)),
                "height": float(np.round(h, 4)),
                "power_density": float(np.round(power, 2))
            })
            
        return blocks

if __name__ == "__main__":
    # Test generation and print formatting
    generator = FloorplanGenerator()
    sample = generator.generate_single_sample(5)
    print(json.dumps(sample, indent=2))
