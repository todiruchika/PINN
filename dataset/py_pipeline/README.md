# Semiconductor Steady-State Thermal Dataset Generator for PINN Projects

This repository hosts a production-grade, multi-threaded CPU thermal simulator designed to generate training, validation, and testing datasets for **Physics-Informed Neural Networks (PINNs)**. It solves the steady-state Poisson heat equation over randomized multi-block semiconductor chip floorplans, extracts sparse sensor readings, renders thermal maps, and validates thermodynamic physical correctness.

---

## 1. Physical and Mathematical Formulations

### Governing Partial Differential Equation
The steady-state temperature field $T(x,y)$ on a 2D thin-film silicon substrate is modeled by the 2D steady-state heat conduction equation (Poisson formulation):

$$-k \nabla^2 T(x, y) = Q(x, y) \quad \text{for } (x, y) \in \Omega$$

Where:
- $\Omega = [0, 1] \times [0, 1]$ is the normalized 2D square semiconductor chip domain.
- $k \in [100.0, 150.0] \text{ W/(m}\cdot\text{K)}$ is the randomized, uniform isotropic **thermal conductivity** of Silicon.
- $T(x, y)$ is the continuous **temperature field** in degrees Celsius ($^\circ\text{C}$).
- $Q(x, y)$ represents the distributed volumetric **power/heat generation profile** in $\text{W/m}^2$, defined as the sum of localized active logic blocks:

$$Q(x, y) = \sum_{i=1}^{M} Q_i \cdot \mathbb{I}_{B_i}(x, y)$$

Where $M \in [1, 10]$ is the number of active heat sources, $Q_i$ is the power density of block $B_i$, and $\mathbb{I}_{B_i}$ is the indicator function of the rectangular block bounds $B_i = [x_{\min}^i, x_{\max}^i] \times [y_{\min}^i, y_{\max}^i] \subset [0,1]^2$.

### Boundary Conditions
We enforce non-homogeneous Dirichlet boundary conditions on the entirety of the spatial boundary $\partial \Omega$:

$$T(x, y) = T_{\text{ambient}} \quad \text{for } (x, y) \in \partial\Omega$$

Where $T_{\text{ambient}}$ is a randomized uniform ambient level representing cooling/sink coupling, satisfying $T_{\text{ambient}} \in [20.0, 40.0]^\circ\text{C}$.

---

## 2. Finite Element Method (FEM) weak form
To solve the PDE using the finite-element solver **DOLFINx**, the governing equations are converted into their corresponding variational (weak) formulation by multiplying with a test function $v \in H^0_1(\Omega)$ and integrating over the domain:

$$\int_{\Omega} k \nabla T \cdot \nabla v \, dx - \int_{\partial\Omega} k (\nabla T \cdot \mathbf{n}) v \, ds = \int_{\Omega} Q v \, dx$$

Since $v = 0$ on the Dirichlet boundary $\partial\Omega$:

$$\text{Find } T \in H^1(\Omega) \text{ s.t. } a(T, v) = L(v) \quad \forall v \in H^0_1(\Omega)$$

Where the bilinear form $a$ and linear form $L$ are:

$$a(T, v) = \int_{\Omega} k \nabla T \cdot \nabla v \, dx$$
$$L(v) = \int_{\Omega} Q v \, dx$$

Discretization is achieved using Lagrange $P_1$ nodal basis functions over a uniform triangular mesh of $128 \times 128$ grid vertices.

---

## 3. Pipeline Directory Architecture

```
dataset/
├── train/
│   ├── images/         # Thermal heatmap images (128x128 PNG, inferno colormap, no axes)
│   ├── matrices/       # Full dense temperature grids (128x128 serialized NumPy .npy files)
│   ├── sensors/        # Sparse sensor readouts (JSON files with positions & temperatures)
│   └── metadata.csv    # Consolidated sample index (columns: ID, hotspots, conductivity, BCs, max_T)
├── val/                # Evaluation split (matches training folder framework)
└── test/               # Independent test split (matches training folder framework)
```

For each sample generated, the coordinates of 5, 8, and 10 sensors are placed randomly, producing local temperature reads.

---

## 4. Quickstart Guide (Local Execution)

The pipeline is written to be executable instantly. If `dolfinx` is not configured locally, the solver resolves the matrices using an extremely fast, high-performance SciPy 2D finite-difference system fallback.

### Prerequisites
Install basic numerical dependencies:
```bash
pip3 install -r requirements.txt
```

### Run Prototype Dataset Generation (700 Samples)
By default, the script generates a dataset split of 500 train, 100 validation, and 100 test samples:
```bash
python3 dataset_generator.py
```

### Scale up to Full PINN Production Scale (60,000 Samples)
Without modifying code, pass the `--scale production` parameter. This leverages Python multiprocess pooling to generate 50,000 training, 5,000 validation, and 5,000 test records in parallel:
```bash
python3 dataset_generator.py --scale production
```

### Parameter Controls (CLI Arguments)
You can customize the generation bounds using CLI flags:
```bash
python3 dataset_generator.py \
  --train-size 1000 \
  --val-size 200 \
  --test-size 200 \
  --grid-size 128 \
  --out-dir custom_dataset \
  --force-fdm
```

---

## 5. Docker Container Orchestration

To run the pipeline inside an isolated virtual container with **full native DOLFINx FEM** bindings fully enabled:

### 1. Build the Docker Image
```bash
docker build -t pinn-thermal-generator:latest .
```

### 2. Run Container and Mount Folder
Generate the prototype dataset and save it directly to a local directories mount on your host machine:
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/workspace/dataset" \
  pinn-thermal-generator:latest
```

### 3. Generate Large-Scale Production Dataset in Docker
```bash
docker run -it --rm \
  -v "$(pwd)/dataset:/workspace/dataset" \
  pinn-thermal-generator:latest --scale production
```

---

## 6. Visualization & Dataset Validation

### 1. Visualize a Specific Sample
```bash
python3 visualization.py \
  --matrix dataset/train/matrices/thermal_00001.npy \
  --sensors dataset/train/sensors/thermal_00001_sensors_5.json \
  --save output_plot.png
```

### 2. Run Dataset Validation Suite
Validates that generated grids have no NaNs, conform to laws of thermal expansion, and ensure heat moves correctly away from localized floorplan elements:
```bash
python3 validate_dataset.py --dir dataset
```

---

## Citation (LaTeX Format)
Excellent for citing this dataset pipeline architecture in research papers:

```latex
@misc{pinn_thermal_dataset_2026,
  author = {Vipul Reddy G.},
  title = {Semiconductor Steady-State Thermal Dataset Generator for PINNs},
  year = {2026},
  publisher = {Google AI Studio Workspace},
  howpublished = {\url{https://ai.studio/build}}
}
```
