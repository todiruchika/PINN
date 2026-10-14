"""Autodiff PDE residual for the HotSpot-style thermal dataset (torch).

The dataset solves (``generator/pde_solver.py``, ``PROTOCOL.md``)

    -div( t_die k(T) grad T ) + h_eff (T - T_amb) = q(x, y, T)      on the die,
    dT/dn = 0                                                       on dOmega,

with ``k(T) = 22714 / (T[K] - 99) + 2`` clipped to [20, 160] W/(m K).

Everything here is written in *normalised* units. With ``x_hat = x / L``,
``theta = (T - T_amb) / S`` for a fixed temperature scale ``S`` and the source
divided by ``h_eff S``, the equation becomes

    -eps div_hat( k(T) grad_hat theta ) + theta = q_tilde ,    eps = t_die / (h_eff L^2)

on the unit square with ``dtheta/dn = 0``. ``k`` stays in W/(m K) because it is
only ever multiplied by the dimensionless ``eps``.

The source ``q_tilde`` is the *total* source (dynamic + leakage at the local
temperature), which is exactly what the dataset stores in ``thermal_NNNNN_Q.npy``.
A model that predicts it directly therefore needs no leakage assumption.

``src/physics/heat.py`` keeps the original numpy stubs; this module is separate so
that it can be used without touching them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

FieldFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class Physics:
    """Constants of the governing equation (defaults = ``generator/pde_solver.py``).

    ``t_die`` and ``h_eff`` are flagged ASSUMPTION -- NOT SOURCED in
    ``PROTOCOL.md``; they are the generator's values, so using them is correct
    for this dataset, but confirm them before publication.

    ``k_const`` replaces ``k(T)`` by a constant; it exists for the unit tests.
    """

    L: float = 0.012                         # die side, m
    t_die: float = 500e-6                    # die thickness, m
    h_eff: float = 1.0 / ((0.01 + 0.1 + 1.0) * 0.012 * 0.012)   # ~6256.3 W/(m^2 K)
    k_A: float = 22714.0
    k_offset: float = 99.0
    k_B: float = 2.0
    k_min: float = 20.0
    k_max: float = 160.0
    k_const: float | None = None

    @property
    def eps(self) -> float:
        """Dimensionless diffusion number ``t_die / (h_eff L^2)`` (about 5.6e-4)."""
        return self.t_die / (self.h_eff * self.L**2)

    def k(self, T_celsius: torch.Tensor) -> torch.Tensor:
        """Silicon conductivity in W/(m K); same clipping as the generator."""
        if self.k_const is not None:
            return torch.full_like(T_celsius, float(self.k_const))
        T_K = torch.clamp(T_celsius + 273.15, 250.0, 1200.0)
        return torch.clamp(self.k_A / (T_K - self.k_offset) + self.k_B, self.k_min, self.k_max)

    def source_scale(self, temp_scale: float) -> float:
        """``h_eff * S``: converts a normalised source to W/m^2."""
        return self.h_eff * float(temp_scale)


def pde_residual(
    fn: FieldFn,
    xy: torch.Tensor,
    T_amb: torch.Tensor,
    physics: Physics,
    temp_scale: float,
) -> torch.Tensor:
    """Strong-form residual ``-eps div(k grad theta) + theta - q_tilde``, shape ``(N, 1)``.

    ``fn(xy)`` returns ``(theta, q_tilde)``, both ``(N, 1)``, with every row
    depending only on its own ``xy`` row (so a batch of chips can be flattened
    into one call). ``T_amb`` is ``(N, 1)`` or a scalar: the ambient temperature
    of the chip each row belongs to, in degC.
    """
    if xy.dim() != 2 or xy.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got {tuple(xy.shape)}")
    xy = xy.clone().requires_grad_(True)
    theta, q = fn(xy)
    if theta.shape[-1] != 1 or q.shape != theta.shape:
        raise ValueError(f"expected two (N, 1) outputs, got {tuple(theta.shape)} and {tuple(q.shape)}")

    grad = torch.autograd.grad(theta.sum(), xy, create_graph=True)[0]
    flux = physics.k(T_amb + temp_scale * theta) * grad                # (N, 2)
    div = 0.0
    for dim in range(2):
        d = torch.autograd.grad(flux[:, dim].sum(), xy, create_graph=True)[0]
        div = div + d[:, dim : dim + 1]
    return -physics.eps * div + theta - q
