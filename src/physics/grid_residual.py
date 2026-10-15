"""Finite-volume residual of the die heat equation on the 128x128 grid.

Equation (PROTOCOL.md, generator/pde_solver.py):

    -div( t_die * k(T) * grad T ) + h_eff * (T - T_amb) = q

Everything here works on the dimensionless variables used by the data-driven
branch: theta = (T - T_amb) / S and q_tilde = q / (h_eff * S), with S = 50 K.
The residual is then

    R = -c * div_h( k(T) grad_h theta ) + theta - q_tilde,
    c = t_die / (h_eff * h^2),   h = DIE_L / (GRID - 1)

The lateral edges are adiabatic. Faces that would point outside the grid carry
zero flux, so the scheme satisfies dT/dn = 0 exactly and the discrete divergence
sums to zero over the grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from generator.pde_solver import (
    DIE_THICKNESS_DEFAULT,
    H_EFF_DEFAULT,
    _K_A,
    _K_B,
    _K_MAX,
    _K_MIN,
    _K_OFFSET,
)

DIE_L = 0.012
GRID = 128
TEMP_SCALE_C = 50.0  # S in the dimensionless variables


def silicon_conductivity(temperature_c: torch.Tensor) -> torch.Tensor:
    """k(T) = 22714 / (T[K] - 99) + 2, clipped to [20, 160] W/(m K)."""
    temperature_k = torch.clamp(temperature_c + 273.15, min=250.0, max=1200.0)
    conductivity = _K_A / (temperature_k - _K_OFFSET) + _K_B
    return torch.clamp(conductivity, min=_K_MIN, max=_K_MAX)


def diffusion_term(theta: torch.Tensor, conductivity: torch.Tensor) -> torch.Tensor:
    """Sum over the 4 neighbours of k_face * (theta_nb - theta_i), zero flux at edges.

    theta and conductivity have shape (B, H, W). The result has the same shape.
    """
    # Face values are the average of the two neighbouring nodes' conductivity.
    k_x = 0.5 * (conductivity[..., 1:] + conductivity[..., :-1])
    k_y = 0.5 * (conductivity[..., 1:, :] + conductivity[..., :-1, :])
    flux_x = k_x * (theta[..., 1:] - theta[..., :-1])          # (B, H, W-1)
    flux_y = k_y * (theta[..., 1:, :] - theta[..., :-1, :])    # (B, H-1, W)

    # Pad with zeros so the edge faces carry no flux (adiabatic boundary).
    flux_x = F.pad(flux_x, (1, 1))                              # (B, H, W+1)
    flux_y = F.pad(flux_y, (0, 0, 1, 1))                        # (B, H+1, W)
    return (flux_x[..., 1:] - flux_x[..., :-1]) + (flux_y[..., 1:, :] - flux_y[..., :-1, :])


def grid_residual(
    theta: torch.Tensor,
    q_tilde: torch.Tensor,
    ambient_c: torch.Tensor,
    t_die: float = DIE_THICKNESS_DEFAULT,
    h_eff: float = H_EFF_DEFAULT,
) -> torch.Tensor:
    """Dimensionless PDE residual R, shape (B, H, W).

    ambient_c has shape (B,) and is the known ambient temperature of each chip.
    """
    h = DIE_L / (GRID - 1)
    c = t_die / (h_eff * h * h)
    temperature_c = ambient_c.view(-1, 1, 1) + TEMP_SCALE_C * theta   # absolute T for k(T)
    conductivity = silicon_conductivity(temperature_c)
    return -c * diffusion_term(theta, conductivity) + theta - q_tilde
