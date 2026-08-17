#!/usr/bin/env python3
"""
Steady-State 2D Heat Conduction PDE Solver
Part of the PINN Thermal Dataset Generation Pipeline.

Solves the PDE: -k * \nabla^2 T = Q(x, y)
Using:
  1. FEniCS/DOLFINx (Finite Element Method - Production FEM solver)
  2. SciPy (Finite Difference Method - High Performance fallback)

Supports arbitrary rectangular heat-producing blocks on a [0,1]x[0,1] grid.
"""

import numpy as np
import os
import sys
from typing import List, Dict, Any, Tuple

# Try to import DOLFINx components if available
try:
    import dolfinx
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    from dolfinx import mesh, fem, io
    from dolfinx.geometry import bounding_volume_tree, compute_collisions_points, compute_colliding_cells
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False

# Try to import SciPy for fast, robust FDM solver
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ThermalPDESolver:
    def __init__(self, grid_size: int = 128, verbose: bool = False):
        self.grid_size = grid_size
        self.verbose = verbose
        if self.verbose:
            print(f"[{'DOLFINx' if DOLFINX_AVAILABLE else 'SciPy FDM Fallback'}] Solver initialized (Grid: {grid_size}x{grid_size})", flush=True)

    def solve_fdm(self, k: float, ambient: float, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Solves the steady-state heat equation using Finite Difference Method (FDM)
        relying on SciPy sparse linear solver.
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required for the fallback FDM solver. Please install scipy.")

        N = self.grid_size
        dx = 1.0 / (N - 1)
        num_nodes = N * N

        # Build Q Map
        Q_map = np.zeros((N, N))
        for block in blocks:
            # Convert physical coordinates [0,1] to indices [0, N-1]
            i_min = max(0, int(np.round(block["y_min"] * (N - 1))))
            i_max = min(N - 1, int(np.round(block["y_max"] * (N - 1))))
            j_min = max(0, int(np.round(block["x_min"] * (N - 1))))
            j_max = min(N - 1, int(np.round(block["x_max"] * (N - 1))))
            
            # Fill block energy density
            Q_map[i_min:i_max+1, j_min:j_max+1] += block["power_density"]

        # Solve system A * T = b
        # Let's create equations for internal grid points, and boundaries
        A = sp.lil_matrix((num_nodes, num_nodes))
        b = np.zeros(num_nodes)

        # Helper function to get index
        idx = lambda i, j: i * N + j

        # We construct five-point stencil for internal points, identity for boundaries
        h2 = dx * dx
        for i in range(N):
            for j in range(N):
                node_idx = idx(i, j)
                if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                    # Dirichlet Boundary Condition
                    A[node_idx, node_idx] = 1.0
                    b[node_idx] = ambient
                else:
                    # -k * (T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j}) / h^2 = Q_{i,j}
                    # T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j} = - (h^2 * Q_{i,j}) / k
                    A[node_idx, idx(i+1, j)] = 1.0
                    A[node_idx, idx(i-1, j)] = 1.0
                    A[node_idx, idx(i, j+1)] = 1.0
                    A[node_idx, idx(i, j-1)] = 1.0
                    A[node_idx, node_idx] = -4.0
                    b[node_idx] = - (h2 * Q_map[i, j]) / k

        # Convert to CSR for high performance solve
        A = A.tocsr()
        T_flat = spla.spsolve(A, b)
        T_matrix = T_flat.reshape((N, N))
        
        return T_matrix

    def solve_fem_dolfinx(self, k: float, ambient: float, blocks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Solves the steady-state heat equation using Finite Element Method (FEM) is DOLFINx
        """
        N = self.grid_size
        
        # Create square mesh (unit coordinates)
        domain = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, N, N, dolfinx.mesh.CellType.triangle
        )
        
        # Define Function Space (Lagrange, 1st order)
        V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
        
        # Define Trial and Test Functions
        T_trial = ufl.TrialFunction(V)
        v_test = ufl.TestFunction(V)
        
        # Set up Dirichlet boundary condition (ksp problem solver requires continuous bound)
        facets = dolfinx.mesh.locate_entities_boundary(
            domain, domain.topology.dim - 1, lambda x: np.full(x.shape[1], True)
        )
        boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)
        T_ambient_const = dolfinx.fem.Constant(domain, PETSc.ScalarType(ambient))
        bc = dolfinx.fem.dirichletbc(T_ambient_const, boundary_dofs, V)
        
        # Set up Source term function Q(x)
        Q = dolfinx.fem.Function(V)
        
        # We can set Q at nodes by measuring coordinates of DOLFINx DOF mapping
        coords = V.tabulate_dof_coordinates()
        q_vals = np.zeros(coords.shape[0])
        
        # Aggregate heat blocks
        for block in blocks:
            # Find points residing inside the bounding boxes
            in_block = (
                (coords[:, 0] >= block["x_min"]) & (coords[:, 0] <= block["x_max"]) &
                (coords[:, 1] >= block["y_min"]) & (coords[:, 1] <= block["y_max"])
            )
            q_vals[in_block] += block["power_density"]
            
        Q.x.array[:] = q_vals
        
        # Define Bilinear component a(T, v) and Linear model L(v)
        k_const = dolfinx.fem.Constant(domain, PETSc.ScalarType(k))
        a = k_const * ufl.inner(ufl.grad(T_trial), ufl.grad(v_test)) * ufl.dx
        L = Q * v_test * ufl.dx
        
        # Set up KSP PETSc problem and solve
        problem = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        uh = problem.solve()
        
        # Interpolate numerical FEM results onto a dense uniform 128x128 array
        x_pts = np.linspace(0.0, 1.0, N)
        y_pts = np.linspace(0.0, 1.0, N)
        X, Y = np.meshgrid(x_pts, y_pts)
        eval_points = np.vstack((X.ravel(), Y.ravel(), np.zeros_like(X.ravel()))).T
        
        bb_tree = bounding_volume_tree(domain, domain.topology.dim)
        cell_cand = compute_collisions_points(bb_tree, eval_points)
        colliding_cells = compute_colliding_cells(domain, cell_cand, eval_points)
        
        cells_list = []
        points_list = []
        for i, pt in enumerate(eval_points):
            links = colliding_cells.links(i)
            if len(links) > 0:
                cells_list.append(links[0])
                points_list.append(pt)
            else:
                cells_list.append(0) # Boundary safety index
                points_list.append(pt)
                
        eval_points_clean = np.array(points_list, dtype=np.float64)
        temps = uh.eval(eval_points_clean, cells_list)
        
        # Reshape to 128x128 grid
        temperature_matrix = temps.reshape((N, N))
        
        return temperature_matrix

    def solve(self, k: float, ambient: float, blocks: List[Dict[str, Any]], force_fdm: bool = False) -> np.ndarray:
        """
        Primary solve coordinator. Automatically chooses DOLFINx if available,
        otherwise falls back seamlessly to the highly optimized SciPy FDM solver.
        """
        if DOLFINX_AVAILABLE and not force_fdm:
            try:
                return self.solve_fem_dolfinx(k, ambient, blocks)
            except Exception as e:
                if self.verbose:
                    print(f"[Solver] Error running DOLFINx solver: {e}. Falling back to SciPy...", flush=True, file=sys.stderr)
                return self.solve_fdm(k, ambient, blocks)
        else:
            return self.solve_fdm(k, ambient, blocks)


if __name__ == "__main__":
    # Test solver
    test_blocks = [
        {"x_min": 0.2, "x_max": 0.5, "y_min": 0.3, "y_max": 0.6, "power_density": 1.5e7}
    ]
    solver = ThermalPDESolver(grid_size=128, verbose=True)
    temp = solver.solve(k=120.0, ambient=30.0, blocks=test_blocks)
    print(f"Simulation completed. Grid min/max temperatures: {temp.min():.2f}°C / {temp.max():.2f}°C")
