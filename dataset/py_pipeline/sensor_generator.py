#!/usr/bin/env python3
"""
Sparse Sensor Generation for PINN Thermal Datasets
Part of the PINN Thermal Dataset Generation Pipeline.

Randomly deploys temperature sensors (5, 8, and 10) inside a 2D chip domain,
extracts local temperatures from the generated temperature matrix, and saves JSON assets.
"""

import numpy as np
import random
import json
from typing import List, Dict, Any, Tuple

class SensorGenerator:
    def __init__(self, grid_size: int = 128):
        self.grid_size = grid_size

    def generate_sensors_for_density(self, temperature_matrix: np.ndarray, count: int) -> Dict[str, Any]:
        """
        Randomly places 'count' sensors on the N x N grid and queries local temperature values.
        """
        N = self.grid_size
        positions = []
        temperatures = []

        # Generate unique coordinate positions
        placed = set()
        while len(placed) < count:
            # Grid index coordinates
            row = random.randint(1, N - 2)  # Avoid exact borders for sensor grids
            col = random.randint(1, N - 2)
            if (row, col) not in placed:
                placed.add((row, col))
                
                # Convert grid indices back to physical dimensions [0.0, 1.0]
                x_phys = float(np.round(col / (N - 1), 6))
                y_phys = float(np.round(row / (N - 1), 6))
                
                positions.append([x_phys, y_phys])
                
                # Fetch temperature value from indices
                # Note: row corresponds to y-coord index, col to x-coord index
                temp_val = float(np.round(temperature_matrix[row, col], 4))
                temperatures.append(temp_val)

        return {
            "sensor_count": count,
            "positions": positions,
            "temperatures": temperatures
        }

    def process_sample(self, temperature_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Generates sensor readouts for all three required densities: 5, 8, and 10 sensors.
        Returns a dictionary grouping all three.
        """
        return {
            "sensors_5": self.generate_sensors_for_density(temperature_matrix, 5),
            "sensors_8": self.generate_sensors_for_density(temperature_matrix, 8),
            "sensors_10": self.generate_sensors_for_density(temperature_matrix, 10)
        }

if __name__ == "__main__":
    # Test sensor generation
    N = 128
    dummy_temp = np.ones((N, N)) * 45.0  # Safe temperature map
    dummy_temp[50:80, 50:80] = 95.0      # Warm hotspot core
    
    generator = SensorGenerator(grid_size=N)
    result = generator.process_sample(dummy_temp)
    print("Consolidated Sensor JSON preview:")
    print(json.dumps(result, indent=2))
