"""Differentiable residuals for the shared steady-state thermal model."""
from __future__ import annotations

import torch

from generator.pde_solver import (
    DIE_THICKNESS_DEFAULT,
    H_EFF_DEFAULT,
    _K_A,
    _K_B,
    _K_MAX,
    _K_MIN,
    _K_OFFSET,
)


def silicon_conductivity_torch(temperature_c: torch.Tensor) -> torch.Tensor:
    """Return the solver's clipped silicon conductivity in W/(m K)."""
    temperature_k = torch.clamp(temperature_c + 273.15, min=250.0, max=1200.0)
    conductivity = _K_A / (temperature_k - _K_OFFSET) + _K_B
    return torch.clamp(conductivity, min=_K_MIN, max=_K_MAX)


def pde_residual(
    temperature_c: torch.Tensor,
    power_density: torch.Tensor,
    ambient_c: torch.Tensor,
    coordinates_m: torch.Tensor,
    t_die: float = DIE_THICKNESS_DEFAULT,
    h_eff: float = H_EFF_DEFAULT,
) -> torch.Tensor:
    """Evaluate ``-div(t*k(T)*grad(T)) + h*(T-T_amb) - Q``.

    Coordinates are physical metres and ``power_density`` is W/m^2. The
    returned residual is in W/m^2 and retains its autograd graph.
    """
    if not coordinates_m.requires_grad:
        raise ValueError("coordinates_m must require gradients")
    if coordinates_m.shape[-1] != 2 or temperature_c.shape != power_density.shape:
        raise ValueError("Expected 2D coordinates and matching temperature/Q shapes")

    temperature_gradient = torch.autograd.grad(
        temperature_c.sum(),
        coordinates_m,
        create_graph=True,
        retain_graph=True,
    )[0]
    conductivity = silicon_conductivity_torch(temperature_c)
    flux_x = conductivity * temperature_gradient[..., 0]
    flux_y = conductivity * temperature_gradient[..., 1]
    flux_x_gradient = torch.autograd.grad(
        flux_x.sum(), coordinates_m, create_graph=True, retain_graph=True
    )[0]
    flux_y_gradient = torch.autograd.grad(
        flux_y.sum(), coordinates_m, create_graph=True, retain_graph=True
    )[0]
    divergence = flux_x_gradient[..., 0] + flux_y_gradient[..., 1]

    ambient = ambient_c
    while ambient.ndim < temperature_c.ndim:
        ambient = ambient.unsqueeze(-1)
    return -t_die * divergence + h_eff * (temperature_c - ambient) - power_density


def neumann_boundary_residual(
    temperature_c: torch.Tensor,
    coordinates_m: torch.Tensor,
    outward_normals: torch.Tensor,
) -> torch.Tensor:
    """Return outward normal temperature gradients on lateral boundaries."""
    if not coordinates_m.requires_grad:
        raise ValueError("coordinates_m must require gradients")
    gradient = torch.autograd.grad(
        temperature_c.sum(),
        coordinates_m,
        create_graph=True,
        retain_graph=True,
    )[0]
    return (gradient * outward_normals).sum(dim=-1)