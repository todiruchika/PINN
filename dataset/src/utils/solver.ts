/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { FloorplanBlock, SensorReadingsGroup, SimulationConfig, SimulationResult } from "../types";

/**
 * Generates an active hardware floorplan for the semiconductor chip.
 */
export function generateRandomFloorplan(hotspotCount: number): FloorplanBlock[] {
  const blockTypes: ("CPU" | "GPU" | "Cache" | "MC")[] = ["CPU", "GPU", "Cache", "MC"];
  const powerDensities = {
    CPU: { min: 1.2e7, max: 2.5e7 },
    GPU: { min: 1.0e7, max: 1.8e7 },
    Cache: { min: 1.0e6, max: 3.0e6 },
    MC: { min: 4.0e6, max: 8.0e6 },
  };

  const blocks: FloorplanBlock[] = [];

  for (let idx = 0; idx < hotspotCount; idx++) {
    const type = blockTypes[Math.floor(Math.random() * blockTypes.length)];
    const pRange = powerDensities[type];
    const power = pRange.min + Math.random() * (pRange.max - pRange.min);

    let w = 0.15;
    let h = 0.15;

    if (type === "CPU") {
      w = 0.12 + Math.random() * 0.13;
      h = 0.12 + Math.random() * 0.13;
    } else if (type === "GPU") {
      w = 0.22 + Math.random() * 0.20;
      h = 0.20 + Math.random() * 0.18;
    } else if (type === "Cache") {
      w = 0.18 + Math.random() * 0.15;
      h = 0.15 + Math.random() * 0.13;
    } else {
      // MC (thin strip near edges)
      const edge = Math.random() < 0.5 ? "horizontal" : "vertical";
      if (edge === "horizontal") {
        w = 0.30 + Math.random() * 0.30;
        h = 0.05 + Math.random() * 0.05;
      } else {
        w = 0.05 + Math.random() * 0.05;
        h = 0.30 + Math.random() * 0.30;
      }
    }

    // Anchor bound positioning [0.0, 1.0 - length]
    const x_min = Math.random() * (1.0 - w);
    const y_min = Math.random() * (1.0 - h);
    const x_max = x_min + w;
    const y_max = y_min + h;

    blocks.push({
      id: `block_${idx}`,
      type,
      x_min: Number(x_min.toFixed(4)),
      y_min: Number(y_min.toFixed(4)),
      x_max: Number(x_max.toFixed(4)),
      y_max: Number(y_max.toFixed(4)),
      width: Number(w.toFixed(4)),
      height: Number(h.toFixed(4)),
      power_density: Math.round(power),
    });
  }

  return blocks;
}

/**
 * 2D Steady-State Heat Equation Solver using Finite Difference Method (SOR)
 * Solves: -k * (d^2 T / dx^2 + d^2 T / dy^2) = Q(x, y)
 * With Boundary Dirichlet conditions: T_boundary = T_ambient
 */
export function solveHeatEquation(
  k: number,
  ambient: number,
  blocks: FloorplanBlock[],
  gridSize: number = 128
): { grid: number[][]; min: number; max: number } {
  const N = gridSize;
  const h = 1.0 / (N - 1);
  const h2 = h * h;

  // Initialize temperature grid with ambient value
  const T = Array.from({ length: N }, () => new Float32Array(N).fill(ambient));

  // Build discretized volumetric heat source profile Q
  const Q = Array.from({ length: N }, () => new Float32Array(N).fill(0));
  blocks.forEach((block) => {
    const i_min = Math.max(0, Math.round(block.y_min * (N - 1)));
    const i_max = Math.min(N - 1, Math.round(block.y_max * (N - 1)));
    const j_min = Math.max(0, Math.round(block.x_min * (N - 1)));
    const j_max = Math.min(N - 1, Math.round(block.x_max * (N - 1)));

    for (let i = i_min; i <= i_max; i++) {
      for (let j = j_min; j <= j_max; j++) {
        Q[i][j] += block.power_density;
      }
    }
  });

  // Successive Over-Relaxation (SOR) iterative solver
  // Optimal relaxation parameter omega for 128 grid size in 2D is ~1.4 - 1.6
  const omega = 1.45;
  const maxIterations = 400;
  const tolerance = 1e-4;

  for (let iter = 0; iter < maxIterations; iter++) {
    let maxDiff = 0;

    for (let i = 1; i < N - 1; i++) {
      for (let j = 1; j < N - 1; j++) {
        // Local FD equation solver component
        const sourceTerm = (h2 * Q[i][j]) / k;
        const targetTemp = 0.25 * (T[i + 1][j] + T[i - 1][j] + T[i][j + 1] + T[i][j - 1] + sourceTerm);

        const diff = targetTemp - T[i][j];
        if (Math.abs(diff) > maxDiff) {
          maxDiff = Math.abs(diff);
        }

        // Apply relaxation step
        T[i][j] += omega * diff;
      }
    }

    if (maxDiff < tolerance) {
      break;
    }
  }

  // Convert Float32Arrays back to standard arrays and retrieve min/max values
  let min = Infinity;
  let max = -Infinity;
  const standardGrid: number[][] = [];

  for (let i = 0; i < N; i++) {
    const row: number[] = [];
    for (let j = 0; j < N; j++) {
      const val = T[i][j];
      row.push(val);
      if (val < min) min = val;
      if (val > max) max = val;
    }
    standardGrid.push(row);
  }

  return { grid: standardGrid, min: Number(min.toFixed(3)), max: Number(max.toFixed(3)) };
}

/**
 * Deploys sparse sensor coordinates and samples local temperature fields from solution grids.
 */
export function extractSensors(grid: number[][], count: number, gridSize: number = 128): {
  sensor_count: number;
  positions: [number, number][];
  temperatures: number[];
} {
  const N = gridSize;
  const positions: [number, number][] = [];
  const temperatures: number[] = [];
  const placed = new Set<string>();

  while (positions.length < count) {
    const row = 3 + Math.floor(Math.random() * (N - 6)); // Avoid exact borders for sensor grids
    const col = 3 + Math.floor(Math.random() * (N - 6));
    const label = `${row},${col}`;

    if (!placed.has(label)) {
      placed.add(label);
      const x_phys = Number((col / (N - 1)).toFixed(5));
      const y_phys = Number((row / (N - 1)).toFixed(5));
      positions.push([x_phys, y_phys]);

      const value = Number(grid[row][col].toFixed(3));
      temperatures.push(value);
    }
  }

  return {
    sensor_count: count,
    positions,
    temperatures,
  };
}

export function extractSensorsGroup(grid: number[][], gridSize: number = 128): SensorReadingsGroup {
  return {
    sensors_5: extractSensors(grid, 5, gridSize),
    sensors_8: extractSensors(grid, 8, gridSize),
    sensors_10: extractSensors(grid, 10, gridSize),
  };
}

/**
 * Realistic color map interpolation (Inferno & Jet)
 */
export function getColormapColor(value: number, min: number, max: number, type: "inferno" | "jet"): string {
  const range = max - min || 1;
  const norm = Math.max(0, Math.min(1, (value - min) / range));

  if (type === "inferno") {
    // Inferno color points (rough approximation for rendering)
    if (norm < 0.25) {
      const p = norm / 0.25;
      const r = Math.round(0 + p * 40);
      const g = Math.round(5 + p * 10);
      const b = Math.round(25 + p * 60);
      return `rgb(${r},${g},${b})`;
    } else if (norm < 0.5) {
      const p = (norm - 0.25) / 0.25;
      const r = Math.round(40 + p * 115);
      const g = Math.round(15 + p * 40);
      const b = Math.round(85 - p * 15);
      return `rgb(${r},${g},${b})`;
    } else if (norm < 0.75) {
      const p = (norm - 0.5) / 0.25;
      const r = Math.round(155 + p * 75);
      const g = Math.round(55 + p * 90);
      const b = Math.round(70 - p * 35);
      return `rgb(${r},${g},${b})`;
    } else {
      const p = (norm - 0.75) / 0.25;
      const r = Math.round(230 + p * 25);
      const g = Math.round(145 + p * 105);
      const b = Math.round(35 + p * 185);
      return `rgb(${r},${g},${b})`;
    }
  } else {
    // Standard Jet color mapping (blue -> cyan -> green -> yellow -> red)
    if (norm < 0.125) {
      const p = norm / 0.125;
      return `rgb(0,0,${Math.round(128 + p * 127)})`; // navy to blue
    } else if (norm < 0.375) {
      const p = (norm - 0.125) / 0.25;
      return `rgb(0,${Math.round(p * 255)},255)`; // blue to cyan
    } else if (norm < 0.625) {
      const p = (norm - 0.375) / 0.25;
      return `rgb(${Math.round(p * 255)},255,${Math.round(255 - p * 255)})`; // cyan to yellow-green
    } else if (norm < 0.875) {
      const p = (norm - 0.625) / 0.25;
      return `rgb(255,${Math.round(255 - p * 255)},0)`; // yellow to red
    } else {
      const p = (norm - 0.875) / 0.125;
      return `rgb(${Math.round(255 - p * 127)},0,0)`; // red to dark red
    }
  }
}
