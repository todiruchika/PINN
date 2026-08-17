/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type BlockType = "CPU" | "GPU" | "Cache" | "MC";

export interface FloorplanBlock {
  id: string;
  type: BlockType;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  width: number;
  height: number;
  power_density: number; // in W/m^2
}

export interface SensorConfig {
  sensor_count: number;
  positions: [number, number][]; // normalized [x, y] coordinates
  temperatures?: number[];        // extracted temperatures
}

export interface SensorReadingsGroup {
  sensors_5: SensorConfig;
  sensors_8: SensorConfig;
  sensors_10: SensorConfig;
}

export interface SimulationConfig {
  k: number;             // thermal conductivity
  ambient: number;       // boundary ambient temperature
  grid_size: number;     // e.g. 128
  hotspot_count: number; // number of random blocks to place
}

export interface SimulationResult {
  sample_id: string;
  blocks: FloorplanBlock[];
  grid: number[][]; // N x N grid of temperatures
  min_temp: number;
  max_temp: number;
  sensors: SensorReadingsGroup;
  k: number;
  ambient: number;
}

export interface PipelineCodeFile {
  name: string;
  filename: string;
  language: string;
  description: string;
  content: string;
}
