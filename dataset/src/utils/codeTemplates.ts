/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { PipelineCodeFile } from "../types";

export const PIPELINE_FILES: PipelineCodeFile[] = [
  {
    name: "PDE Finite Element Solver",
    filename: "pde_solver.py",
    language: "python",
    description: "Computes temperature grids by solving steady-state isotropic heat equation using modern FEniCS (DOLFINx). Incorporates a high-performance fallback using SciPy sparse math arrays.",
    content: "python3 pde_solver.py"
  },
  {
    name: "Dataset Master Orchestrator",
    filename: "dataset_generator.py",
    language: "python",
    description: "The primary pipeline engine. Spawns multiprocessing pools to generate floorplans, run thermal solvers, extract sensor datasets, compile csv indices, and split partitions.",
    content: "python3 dataset_generator.py --scale prototype"
  },
  {
    name: "Floorplan Architecture Generator",
    filename: "floorplan_generator.py",
    language: "python",
    description: "Randomly deploys typical computer hardware blocks (CPU cores, GPUs, SRAM stacks, and edge Memory Controllers) onto the chip coordinate spectrum.",
    content: "python3 floorplan_generator.py"
  },
  {
    name: "Sparse Sensor Generator",
    filename: "sensor_generator.py",
    language: "python",
    description: "Places arrays of 5, 8, and 10 temperature sensors across the substrate, measuring local temperature values on the solved grid.",
    content: "python3 sensor_generator.py"
  },
  {
    name: "Thermal Dataset Validator",
    filename: "validate_dataset.py",
    language: "python",
    description: "Evaluates physical grid bounds, checks for numerical anomalies (such as NaNs, Nulls, or Infs), forces thermal boundaries, and validates diffuse energy propagation.",
    content: "python3 validate_dataset.py --dir dataset"
  },
  {
    name: "Scientific Visualizer",
    filename: "visualization.py",
    language: "python",
    description: "Loads temperature matrices and sensor locations, and displays high-resolution thermal heatmaps overlaid with chip layouts and sensor readouts.",
    content: "python3 visualization.py --matrix dataset/train/matrices/thermal_00001.npy"
  },
  {
    name: "Docker Container Structure",
    filename: "Dockerfile",
    language: "dockerfile",
    description: "Integrates the official FEniCSx DOLFINx image, pre-packaged with MPI bindings and PETSc solvers. Configures scientific dependencies with quick caching mechanics.",
    content: "docker build -t pinn-thermal-generator .\ndocker run -v $(pwd)/dataset:/workspace/dataset pinn-thermal-generator"
  },
  {
    name: "Python Requirements Configuration",
    filename: "requirements.txt",
    language: "plaintext",
    description: "Core academic stack requirements specifying exact stable versions of numpy, pandas, matplotlib, and scipy for standard executions.",
    content: "pip3 install -r requirements.txt"
  }
];
