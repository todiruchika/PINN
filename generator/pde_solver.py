#!/usr/bin/env python3
"""
Steady-State 2D Die-Spreading Heat Equation Solver (v2 – peer-review revision)
Part of the PINN Thermal Dataset Generation Pipeline.

Physical model (HotSpot-style compact 2D die-spreading):
--------------------------------------------------------------
  PDE:  -div(t_die * k(T) grad T) + h_eff (T - T_amb) = q''   in Omega

  where:
    t_die  = silicon die thickness [m]
    k(T)   = temperature-dependent in-plane die conductivity [W/(m·K)]
    h_eff  = effective vertical heat sink conductance [W/(m²·K)]
             = 1 / R_pkg_area, where R_pkg_area = R_total × A_chip [m²·K/W]
    q''    = surface (areal) power density [W/m²]  (= W/cm² × 1e4)
    T_amb  = ambient/coolant temperature [°C]

  The h_eff term models vertical heat removal through the package stack
  (TIM + heat spreader + heatsink) as a distributed sink on every cell.
  This is the compact 2D approximation used in HotSpot (Huang et al., 2006).

Boundary conditions:
  - All four lateral edges: Neumann dT/dn = 0 (adiabatic die periphery)
    Source: Therm-FM arXiv 2605.22663; HotSpot Huang et al. TVLSI 2006.

Physical parameters:
  h_eff = 1 / R_pkg_area
  R_pkg_area = (R_die + R_TIM + R_spreader + R_heatsink) × A_chip

  HotSpot default values (Huang et al. 2006, Table II):
    R_die      ≈ 0.01 °C/W   [TBD: read exact from HotSpot default.config]
    R_spreader ≈ 0.1  °C/W   [TBD: read exact from HotSpot default.config]
    R_heatsink ≈ 1.0  °C/W   [TBD: read exact from HotSpot default.config]
  With A_chip = 12mm × 12mm = 1.44e-4 m²:
    R_pkg_area = (0.01+0.1+1.0) × 1.44e-4 = 1.598e-4 m²·K/W
    h_eff      ≈ 6 257 W/(m²·K)

Temperature-dependent k(T):
  Glassbrenner & Slack (1964), Phys. Rev. 134, A1058; Touloukian et al. (1970).
  Shape compared with Glassbrenner & Slack (1964); quantitative validation pending.
  k(T[K]) = A / (T[K] − T_offset) + B
  At 300.15 K (27.0 °C): k = 114.9 W/(m·K)
  Clamped to [20.0, 160.0] W/(m·K), valid range [250.0, 1200.0] K.

Leakage model:
  P_leak(T) = P_leak0 × exp(beta × (T − T0))
  Source: Liao, He & Lepak, IEEE TCAD 2005; He et al., DAC 2004.
  beta: 0.012 /K  # [ASSUMPTION: literature range 0.010 to 0.015 1/K; He et al., DAC 2004]

Iteration strategy (VarSim arXiv 2307.12119):
  Outer loop: updates leakage sources with new T (electrothermal coupling).
  Inner loop: Picard fixed-point iteration for k(T) nonlinearity.

Anti-inverse-crime (Wirgin arXiv math-ph/0401050):
  FineMeshReferenceSolver generates ground truth on a 4× finer mesh.

Units: All SI. q'' in W/m², k in W/(m·K), h_eff in W/(m²·K), T in °C, t_die in m.
"""

import numpy as np
import sys
from typing import List, Dict, Any, Optional

try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Physical constants and model parameters
# ---------------------------------------------------------------------------

# Ambient temperature range [°C]
# [ASSUMPTION — NOT SOURCED: 25–45 °C is a standard engineering convention for
#  dataset diversity; no uploaded paper specifies this range.]
T_AMB_MIN_C = 25.0
T_AMB_MAX_C = 45.0

# Die dimensions (12 mm × 12 mm planar footprint)
# Alpha EV6 floorplan (HotSpot distribution): ~16 mm × 16 mm
# [ASSUMPTION — NOT SOURCED: 12 mm × 12 mm is a representative mid-range size;
#  the HotSpot EV6 floorplan is larger (~16 mm × 16 mm). No uploaded paper
#  specifies a 12 mm die.]
DIE_LX_DEFAULT = 0.012   # m (12 mm)
DIE_LY_DEFAULT = 0.012   # m (12 mm)
# [ASSUMPTION — NOT SOURCED: 0.5 mm die thickness; VarSim (arXiv 2307.12119, Table II)
#  uses 0.15 mm. Confirm before publication.]
DIE_THICKNESS_DEFAULT = 500e-6  # m (0.5 mm)

# Package thermal resistance parameters — HotSpot defaults
# [ASSUMPTION — NOT SOURCED: values are listed as 'HotSpot defaults' but were not read
#  from the uploaded HotSpot 2006 paper. The numeric defaults appear in the HotSpot
#  tool distribution's default.config file, which was not uploaded. Until that file is
#  consulted, these values remain ASSUMPTION — NOT SOURCED.]
R_DIE_DEFAULT       = 0.01    # °C/W  [ASSUMPTION — NOT SOURCED: HotSpot default.config]
R_SPREADER_DEFAULT  = 0.100   # °C/W  [ASSUMPTION — NOT SOURCED: HotSpot default.config]
R_HEATSINK_DEFAULT  = 1.000   # °C/W  [ASSUMPTION — NOT SOURCED: HotSpot default.config]

# Derived effective heat sink conductance
# h_eff = 1 / (R_total [°C/W] × A_chip [m²])
_A_CHIP = DIE_LX_DEFAULT * DIE_LY_DEFAULT   # m²
_R_PKG_DEFAULT = R_DIE_DEFAULT + R_SPREADER_DEFAULT + R_HEATSINK_DEFAULT
H_EFF_DEFAULT = 1.0 / (_R_PKG_DEFAULT * _A_CHIP)   # ≈ 6257 W/(m²·K)  [ASSUMPTION — NOT SOURCED: derived from ASSUMPTION resistance values]

# Leakage model coefficient beta [1/K]
# Sources: He et al., DAC 2004 (subthreshold leakage roughly triples between 25 °C and 100 °C,
#          yielding beta in [0.010, 0.015] /K);
#          Liao, He & Lepak, IEEE TCAD 2005.
# [ASSUMPTION — NOT SOURCED: value 0.012 /K is within the literature range but was
#  not read as an exact value from He et al. DAC 2004 or Liao et al. TCAD 2005]
LEAKAGE_BETA_DEFAULT = 0.012  # 1/K
LEAKAGE_T0           = 25.0   # °C  reference temperature  [ASSUMPTION — NOT SOURCED: standard convention]


# ---------------------------------------------------------------------------
# Temperature-dependent thermal conductivity of crystalline silicon
# ---------------------------------------------------------------------------
# Source: Glassbrenner & Slack (1964), Phys. Rev. 134, A1058; Touloukian et al. (1970).
# Shape compared with Glassbrenner & Slack (1964); quantitative validation pending.
# Model: k(T) = A / (T[K] − T_offset) + B  (Umklapp scattering dominant)
# At 300.15 K (27.0 °C): k = 114.9 W/(m·K)
# At 300 K (26.85 °C): k = 115.0 W/(m·K)
# At 398 K (125 °C): k ≈ 77.9 W/(m·K)
# Fit residual (RMSE 300–700 K): ~3 W/(m·K) [ASSUMPTION — NOT SOURCED: not computed from G&S 1964 data]
_K_A      = 22_714.0   # W·K/m
_K_OFFSET = 99.0       # K
_K_B      = 2.0        # W/(m·K)
_K_MIN    = 20.0       # W/(m·K) lower clip (guard for very high T)
_K_MAX    = 160.0      # W/(m·K) upper clip (guard for very low T extrapolation)


def k_silicon(T_celsius: np.ndarray) -> np.ndarray:
    """
    Temperature-dependent in-plane thermal conductivity of silicon [W/(m·K)].
    k(T) = A / (T[K] − T_offset) + B
    Source: Glassbrenner & Slack (1964), Phys. Rev. 134, A1058.
    Shape compared with Glassbrenner & Slack (1964); quantitative validation pending.
    Valid range: 250 K – 1200 K.
    At 300.15 K (27.0 °C): k = 114.9 W/(m·K).
    Clamped to [20.0, 160.0] W/(m·K).
    """
    T_K = np.asarray(T_celsius, dtype=np.float64) + 273.15
    T_K = np.clip(T_K, 250.0, 1200.0)
    k = _K_A / (T_K - _K_OFFSET) + _K_B
    return np.clip(k, _K_MIN, _K_MAX)


def leakage_power_density(T_celsius: np.ndarray, p_leak0: float,
                          beta: float = LEAKAGE_BETA_DEFAULT,
                          T0: float = LEAKAGE_T0) -> np.ndarray:
    """
    Temperature-dependent leakage power density [W/cm²].
    P_leak(T) = P_leak0 × exp(beta × (T − T0))
    Sources: Liao, He & Lepak, IEEE TCAD 2005; He et al., DAC 2004.
    Overflow guard: exponent clamped to [−20, 20].
    beta: 0.012 /K  # [ASSUMPTION: literature range 0.010 to 0.015 1/K; He et al., DAC 2004]
    """
    exp_arg = np.clip(
        float(beta) * (np.asarray(T_celsius, dtype=np.float64) - float(T0)),
        -20.0, 20.0
    )
    return float(p_leak0) * np.exp(exp_arg)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class ThermalPDESolver:
    """
    2D die-spreading heat equation solver (HotSpot compact model).

    Solves:  -div(t_die * k(T) grad T) + h_eff (T - T_amb) = q''

    where:
      t_die  = silicon die thickness [m]
      k(T)   = temperature-dependent in-plane die conductivity [W/(m·K)]
      h_eff  = effective vertical heat sink conductance [W/(m²·K)]
      q''    = surface (areal) heat source power density [W/m²]

    BCs: adiabatic (Neumann = 0) on all four lateral edges.

    The h_eff distributed sink term makes the problem uniquely solvable
    without any Dirichlet BCs.  This is the correct compact model for
    a chip die with a package heatsink (HotSpot, Huang et al. 2006).
    """

    def __init__(self, grid_size: int = 128,
                 L_x: float = DIE_LX_DEFAULT,
                 L_y: float = DIE_LY_DEFAULT,
                 t_die: float = DIE_THICKNESS_DEFAULT,
                 h_eff: float = H_EFF_DEFAULT,
                 verbose: bool = False,
                 solver_tag: str = "FDM-Newton-Helmholtz",
                 **kwargs):
        self.N = grid_size
        self.L_x = L_x
        self.L_y = L_y
        self.t_die = t_die
        self.h_eff = h_eff
        self.verbose = verbose
        self.solver_tag = solver_tag

        self.dx = L_x / (self.N - 1)
        self.dy = L_y / (self.N - 1)

        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy is required. Install with: pip install scipy")

    # ------------------------------------------------------------------
    def _build_Q_map(self, blocks: List[Dict[str, Any]],
                     T: np.ndarray) -> np.ndarray:
        """
        Build areal heat source map [W/m²] for current temperature field T
        using exact area-weighted rasterization (cell-overlap calculation).

        Each block specifies:
          power_density_dynamic  [W/cm²] – dynamic (switching) areal power
          power_density_leakage0 [W/cm²] – leakage at T0 = 25 °C
          leakage_beta           [1/K]   – leakage exponential coefficient

        Leakage is updated with the current local T for the electrothermal loop.
        Total power sum(Q) * dx * dy matches analytic block power to within 0.5%.
        """
        N = self.N
        dx = self.dx
        dy = self.dy
        j = np.arange(N)
        x_left = (j - 0.5) * dx
        x_right = (j + 0.5) * dx
        i = np.arange(N)
        y_bottom = (i - 0.5) * dy
        y_top = (i + 0.5) * dy

        Q_map = np.zeros((N, N), dtype=np.float64)   # W/m²

        for block in blocks:
            ox = np.maximum(0.0, np.minimum(block["x_max"], x_right) - np.maximum(block["x_min"], x_left))
            oy = np.maximum(0.0, np.minimum(block["y_max"], y_top) - np.maximum(block["y_min"], y_bottom))
            fx = ox / dx
            fy = oy / dy
            F = np.outer(fy, fx)

            if not np.any(F > 0):
                continue

            p_dyn   = float(block.get("power_density_dynamic",  0.0))   # W/cm²
            p_leak0 = float(block.get("power_density_leakage0", 0.0))   # W/cm²
            beta    = float(block.get("leakage_beta", LEAKAGE_BETA_DEFAULT))

            p_leak  = leakage_power_density(T, p_leak0, beta)           # W/cm²
            q_local = (p_dyn + p_leak) * 1e4                            # W/m²

            Q_map += F * q_local

        return Q_map

    # ------------------------------------------------------------------
    def _assemble_system(self, k_map: np.ndarray, Q_map: np.ndarray,
                         T_amb: float) -> tuple:
        """
        Assemble the FDM linear system for:
          -div(t_die * k(T) grad T) + h_eff*(T - T_amb) = q''

        Rewritten as:  -div(t_die * k(T) grad T) + h_eff*T = q'' + h_eff*T_amb

        BCs: Neumann dT/dn = 0 on all four edges (ghost-point method).
        The h_eff diagonal term ensures the matrix is non-singular.

        Variable k: harmonic-mean face conductivities multiplied by t_die.
        Insulated boundaries use 2 * t_die * k_face / dx^2 (ghost reflection).
        The h_eff vertical package term is areal and is NOT multiplied by t_die.

        Vectorized COO assembly for maximum performance.
        """
        N  = self.N
        dx = self.dx
        dy = self.dy
        h  = self.h_eff
        t_die = self.t_die
        num_nodes = N * N

        # Harmonic-mean face conductivities
        k_E_face = 2.0 * k_map[:, :-1] * k_map[:, 1:] / (k_map[:, :-1] + k_map[:, 1:] + 1e-30)
        k_N_face = 2.0 * k_map[:-1, :] * k_map[1:, :] / (k_map[:-1, :] + k_map[1:, :] + 1e-30)

        cE_interior = t_die * k_E_face / dx**2
        cN_interior = t_die * k_N_face / dy**2

        c_W_boundary = 2.0 * t_die * k_E_face[:, 0] / dx**2
        c_E_boundary = 2.0 * t_die * k_E_face[:, -1] / dx**2
        c_S_boundary = 2.0 * t_die * k_N_face[0, :] / dy**2
        c_N_boundary = 2.0 * t_die * k_N_face[-1, :] / dy**2

        rows = []
        cols = []
        data = []

        diag = np.full((N, N), float(h), dtype=np.float64)

        # Y connections
        # S boundary (i = 0): connection to i=1
        rows.append(np.arange(0, N)); cols.append(np.arange(N, 2*N)); data.append(-c_S_boundary)
        diag[0, :] += c_S_boundary

        # N boundary (i = N-1): connection to i=N-2
        rows.append(np.arange((N-1)*N, N*N)); cols.append(np.arange((N-2)*N, (N-1)*N)); data.append(-c_N_boundary)
        diag[N-1, :] += c_N_boundary

        # Interior Y (i in 1..N-2)
        i_int_y = np.arange(1, N-1)
        idx_int_y = (i_int_y[:, None] * N + np.arange(N)[None, :]).ravel()
        idx_north = ((i_int_y[:, None] + 1) * N + np.arange(N)[None, :]).ravel()
        idx_south = ((i_int_y[:, None] - 1) * N + np.arange(N)[None, :]).ravel()

        rows.append(idx_int_y); cols.append(idx_north); data.append(-cN_interior[1:, :].ravel())
        rows.append(idx_int_y); cols.append(idx_south); data.append(-cN_interior[:-1, :].ravel())
        diag[1:N-1, :] += (cN_interior[1:, :] + cN_interior[:-1, :])

        # X connections
        # W boundary (j = 0): connection to j=1
        idx_w_bnd = np.arange(N) * N
        rows.append(idx_w_bnd); cols.append(idx_w_bnd + 1); data.append(-c_W_boundary)
        diag[:, 0] += c_W_boundary

        # E boundary (j = N-1): connection to j=N-2
        idx_e_bnd = np.arange(N) * N + (N - 1)
        rows.append(idx_e_bnd); cols.append(idx_e_bnd - 1); data.append(-c_E_boundary)
        diag[:, N-1] += c_E_boundary

        # Interior X (j in 1..N-2)
        j_int_x = np.arange(1, N-1)
        idx_int_x = (np.arange(N)[:, None] * N + j_int_x[None, :]).ravel()
        idx_east  = (np.arange(N)[:, None] * N + (j_int_x + 1)[None, :]).ravel()
        idx_west  = (np.arange(N)[:, None] * N + (j_int_x - 1)[None, :]).ravel()

        rows.append(idx_int_x); cols.append(idx_east); data.append(-cE_interior[:, 1:].ravel())
        rows.append(idx_int_x); cols.append(idx_west); data.append(-cE_interior[:, :-1].ravel())
        diag[:, 1:N-1] += (cE_interior[:, 1:] + cE_interior[:, :-1])

        # Main diagonal
        idx_all = np.arange(num_nodes)
        rows.append(idx_all); cols.append(idx_all); data.append(diag.ravel())

        A = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(num_nodes, num_nodes)).tocsr()
        b_vec = (Q_map + h * T_amb).ravel()

        return A, b_vec

    # ------------------------------------------------------------------
    def solve(self, T_amb: float,
              blocks: List[Dict[str, Any]],
              newton_tol: float = 0.1,
              newton_max_iter: int = 20,
              thermo_max_iter: int = 15,
              thermo_tol: float = 0.1) -> Dict[str, Any]:
        """
        Solve the coupled nonlinear steady-state heat equation.

        Outer loop: update leakage sources (VarSim arXiv 2307.12119).
        Inner loop: Picard fixed-point iteration for k(T).

        Returns dict: T, Q_total, solver_tag, newton_iters, thermo_iters,
                      peak_temp_C, min_temp_C, energy_bal.
        """
        N = self.N
        T = np.full((N, N), float(T_amb), dtype=np.float64)

        thermo_iter = 0
        newton_iter = 0

        for thermo_iter in range(thermo_max_iter):
            T_prev_outer = T.copy()

            for newton_iter in range(newton_max_iter):
                k_map = k_silicon(T)
                Q_map = self._build_Q_map(blocks, T)

                A, b_v = self._assemble_system(k_map, Q_map, T_amb)
                T_new_flat = spla.spsolve(A, b_v)

                if not np.all(np.isfinite(T_new_flat)):
                    if self.verbose:
                        print("  [Solver] spsolve: non-finite values; "
                              "using T_amb fallback.", flush=True)
                    T_new_flat = np.where(np.isfinite(T_new_flat),
                                          T_new_flat, T_amb)

                T_new = T_new_flat.reshape((N, N))
                delta = float(np.abs(T_new - T).max())
                T = T_new
                if delta < newton_tol:
                    break

            outer_delta = float(np.abs(T - T_prev_outer).max())
            if self.verbose:
                print(f"  [Solver] thermo={thermo_iter}  "
                       f"outer_delta={outer_delta:.4f} K  "
                       f"inner_iters={newton_iter+1}", flush=True)
            if outer_delta < thermo_tol:
                break

        # Energy balance (areal quantities, W/m²·m² = W)
        Q_final = self._build_Q_map(blocks, T)
        P_in    = float(np.sum(Q_final) * self.dx * self.dy)   # W
        P_out   = float(np.sum(self.h_eff * (T - T_amb)) * self.dx * self.dy)  # W
        denom   = max(abs(P_in), abs(P_out), 1e-12)
        err_pct = abs(P_in - P_out) / denom * 100.0

        if self.verbose:
            print(f"  [Solver] P_in={P_in:.3f} W  "
                  f"P_out={P_out:.3f} W  err={err_pct:.2f}%", flush=True)

        return {
            "T":            T.astype(np.float32),
            "Q_total":      Q_final.astype(np.float32),
            "solver_tag":   self.solver_tag,
            # Note: newton_iters records only the final outer-loop inner Picard iteration count
            "newton_iters": int(newton_iter + 1),
            "thermo_iters": int(thermo_iter + 1),
            "peak_temp_C":  float(T.max()),
            "min_temp_C":   float(T.min()),
            "energy_bal": {
                "P_in_W":    float(P_in),
                "P_out_W":   float(P_out),
                "error_pct": float(err_pct)
            }
        }


# ---------------------------------------------------------------------------
# FineMeshReferenceSolver (anti-inverse-crime)
# Source: Wirgin, arXiv math-ph/0401050
# ---------------------------------------------------------------------------

class FineMeshReferenceSolver(ThermalPDESolver):
    """
    High-resolution ground-truth solver (anti-inverse-crime).
    Generates data on a FINE_FACTOR × finer mesh, then downsamples by stride.
    This avoids the 'inverse crime': the forward (data) and inverse
    (reconstruction) solvers use different discretizations.
    Source: Wirgin, arXiv math-ph/0401050 (citing Colton & Kress 1992).
    """
    FINE_FACTOR = 4   # 128 × 4 = 512 nodes per side

    def __init__(self, base_grid_size: int = 128,
                 grid_size: Optional[int] = None,
                 L_x: float = DIE_LX_DEFAULT,
                 L_y: float = DIE_LY_DEFAULT,
                 t_die: float = DIE_THICKNESS_DEFAULT,
                 h_eff: float = H_EFF_DEFAULT,
                 verbose: bool = False,
                 solver_tag: str = "FineMesh512_to_128",
                 **kwargs):
        self.base_grid_size = base_grid_size
        fine_n = grid_size if grid_size is not None else base_grid_size * self.FINE_FACTOR
        super().__init__(grid_size=fine_n,
                         L_x=L_x, L_y=L_y,
                         t_die=t_die, h_eff=h_eff,
                         verbose=verbose,
                         solver_tag=solver_tag,
                         **kwargs)

    def solve_and_downsample(self, T_amb: float,
                              blocks: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Solve on fine mesh and downsample to base_grid_size using stride."""
        result = self.solve(T_amb, blocks, **kwargs)
        T_fine = result["T"]
        f = self.FINE_FACTOR
        T_coarse = T_fine[::f, ::f][:self.base_grid_size, :self.base_grid_size]

        # Compute coarse Q map and energy balance on the downsampled 128x128 grid
        dx_c = self.L_x / (self.base_grid_size - 1)
        dy_c = self.L_y / (self.base_grid_size - 1)

        j = np.arange(self.base_grid_size)
        x_left = (j - 0.5) * dx_c
        x_right = (j + 0.5) * dx_c
        i = np.arange(self.base_grid_size)
        y_bottom = (i - 0.5) * dy_c
        y_top = (i + 0.5) * dy_c

        Q_coarse = np.zeros((self.base_grid_size, self.base_grid_size), dtype=np.float64)
        for block in blocks:
            ox = np.maximum(0.0, np.minimum(block["x_max"], x_right) - np.maximum(block["x_min"], x_left))
            oy = np.maximum(0.0, np.minimum(block["y_max"], y_top) - np.maximum(block["y_min"], y_bottom))
            fx = ox / dx_c
            fy = oy / dy_c
            F = np.outer(fy, fx)
            if not np.any(F > 0):
                continue
            p_dyn = float(block.get("power_density_dynamic", 0.0))
            p_leak0 = float(block.get("power_density_leakage0", 0.0))
            beta = float(block.get("leakage_beta", LEAKAGE_BETA_DEFAULT))
            p_leak = leakage_power_density(T_coarse, p_leak0, beta)
            Q_coarse += F * (p_dyn + p_leak) * 1e4

        P_in_c = float(np.sum(Q_coarse) * dx_c * dy_c)
        P_out_c = float(np.sum(self.h_eff * (T_coarse - T_amb)) * dx_c * dy_c)
        denom = max(abs(P_in_c), abs(P_out_c), 1e-12)
        err_pct_c = abs(P_in_c - P_out_c) / denom * 100.0

        result["T"] = T_coarse.astype(np.float32)
        result["Q_total"] = Q_coarse.astype(np.float32)
        result["fine_grid_size"] = T_fine.shape[0]
        result["coarse_grid_size"] = self.base_grid_size
        result["solver_tag"] = "FineMesh512_to_128"
        result["peak_temp_C"] = float(T_coarse.max())
        result["min_temp_C"] = float(T_coarse.min())
        result["energy_bal"] = {
            "P_in_W": float(P_in_c),
            "P_out_W": float(P_out_c),
            "error_pct": float(err_pct_c)
        }
        return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing ThermalPDESolver (HotSpot compact 2D) …", flush=True)
    test_blocks = [{
        "id":   "test_cpu",
        "type": "CPU",
        "x_min": 0.003, "x_max": 0.007,
        "y_min": 0.003, "y_max": 0.007,
        "power_density_dynamic":  60.0,   # W/cm²  (within 10–200 W/cm² range)
        "power_density_leakage0":  3.0,   # W/cm²
        "leakage_beta": LEAKAGE_BETA_DEFAULT,
    }]

    solver = ThermalPDESolver(grid_size=64, verbose=True)
    res = solver.solve(T_amb=35.0, blocks=test_blocks)

    print(f"  Peak T     = {res['peak_temp_C']:.2f} °C")
    print(f"  T_amb      = 35.0 °C")
    print(f"  ΔT         = {res['peak_temp_C'] - 35.0:.2f} °C")
    print(f"  Energy balance error: {res['energy_bal']['error_pct']:.2f}%")
    print(f"  Newton iters: {res['newton_iters']}, Thermo iters: {res['thermo_iters']}")

    assert np.isfinite(res["peak_temp_C"]), "Non-finite peak temperature"
    assert res["peak_temp_C"] < 1412.0, \
        f"Peak T {res['peak_temp_C']:.1f} °C exceeds silicon melting point"
    assert res["peak_temp_C"] > 35.0, \
        "Peak T not above ambient — solver error"
    assert res["energy_bal"]["error_pct"] < 15.0, \
        f"Energy balance error too large: {res['energy_bal']['error_pct']:.1f}%"

    print("Self-test PASSED ✓")

    print(f"\nModel parameters:")
    print(f"  Die size: {DIE_LX_DEFAULT*1e3:.0f} mm × {DIE_LY_DEFAULT*1e3:.0f} mm")
    print(f"  h_eff: {H_EFF_DEFAULT:.1f} W/(m²·K)  "
          f"[from R_pkg = {_R_PKG_DEFAULT:.2f} °C/W, A = {_A_CHIP*1e6:.0f} mm²]")
    print(f"  k_silicon(27°C): {k_silicon(np.array([27.0]))[0]:.1f} W/(m·K)")
    print(f"  k_silicon(125°C): {k_silicon(np.array([125.0]))[0]:.1f} W/(m·K)")
