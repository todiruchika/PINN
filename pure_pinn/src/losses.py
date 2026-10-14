"""PDE residual and data/boundary losses for the HotSpot-style thermal dataset.

Shared by ``models/pinn.py`` (Model B) and, later, ``models/hybrid.py``
(Model C) so the physics is written once and validated once.

The dataset solves (``generator/pde_solver.py``, ``PROTOCOL.md``)

    -div( t_die k(T) grad T ) + h_eff (T - T_amb) = q(x, y, T)      on the die,
    dT/dn = 0                                                       on dOmega,

with ``k(T) = 22714 / (T[K] - 99) + 2`` clipped to [20, 160] W/(m K) and a
leakage source ``q = (p_dyn + p_leak0 exp(beta (T - T0))) * 1e4`` W/m^2.

Everything below is written in *normalised* units. With ``x_hat = x / L``,
``theta = (T - T_amb) / S`` (``S`` a per-chip temperature scale) and the source
divided by ``h_eff S``, the equation becomes

    -eps div_hat( k(T) grad_hat theta ) + theta = q_tilde ,    eps = t_die / (h_eff L^2)

on the unit square with ``dtheta/dn = 0``. ``k`` stays in W/(m K) because it is
only ever multiplied by the dimensionless ``eps``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
# physical constants (mirrors generator/pde_solver.py)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Physics:
    """Constants of the governing equation.

    Defaults equal the values in ``generator/pde_solver.py``. ``t_die``,
    ``h_eff`` and ``beta`` are flagged ASSUMPTION -- NOT SOURCED in
    ``PROTOCOL.md``; they are the generator's values, so using them is correct
    for this dataset but they must be confirmed before publication.

    ``leak_fraction`` is the one quantity the solver does not expose as a
    constant: each block's leakage at ``T0`` is ``p_dyn`` times a per-type
    fraction (CPU 0.05, GPU 0.05, Cache 0.04, MC 0.03 in
    ``floorplan_generator.py``). A physics-only model does not know the block
    types, so it uses their mean. Set it to ``0`` to drop leakage entirely.

    ``k_const`` replaces ``k(T)`` by a constant; it exists for the unit tests.
    """

    L: float = 0.012                         # die side, m
    t_die: float = 500e-6                    # die thickness, m
    h_eff: float = 1.0 / ((0.01 + 0.1 + 1.0) * 0.012 * 0.012)   # ~6256.3 W/(m^2 K)
    beta: float = 0.012                      # leakage coefficient, 1/K
    T0: float = 25.0                         # leakage reference temperature, degC
    leak_fraction: float = 0.0425            # mean p_leak0 / p_dyn over block types
    k_A: float = 22714.0
    k_offset: float = 99.0
    k_B: float = 2.0
    k_min: float = 20.0
    k_max: float = 160.0
    k_const: float | None = None

    @property
    def eps(self) -> float:
        """Dimensionless diffusion number ``t_die / (h_eff L^2)``."""
        return self.t_die / (self.h_eff * self.L**2)

    def k(self, T_celsius: torch.Tensor) -> torch.Tensor:
        """Silicon conductivity in W/(m K); same clipping as the generator."""
        if self.k_const is not None:
            return torch.full_like(T_celsius, float(self.k_const))
        T_K = torch.clamp(T_celsius + 273.15, 250.0, 1200.0)
        return torch.clamp(self.k_A / (T_K - self.k_offset) + self.k_B, self.k_min, self.k_max)

    def leakage_factor(self, T_celsius: torch.Tensor) -> torch.Tensor:
        """``1 + f * exp(beta (T - T0))``: total source over dynamic source."""
        arg = torch.clamp(self.beta * (T_celsius - self.T0), -20.0, 20.0)
        return 1.0 + self.leak_fraction * torch.exp(arg)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "Physics":
        return cls(**(cfg or {}))


@dataclass(frozen=True)
class ChipContext:
    """The per-chip constants that turn ``theta`` back into degC."""

    T_amb: float
    scale: float

    def to_celsius(self, theta: torch.Tensor) -> torch.Tensor:
        return self.T_amb + self.scale * theta


# --------------------------------------------------------------------------- #
# differential operators
# --------------------------------------------------------------------------- #
def laplacian(fn: TensorFn, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(value, laplacian)`` of a scalar field at ``xy`` via autodiff.

    ``xy`` is ``(N, 2)``; the returned tensors are ``(N, 1)``. The graph is
    retained so the result can be differentiated w.r.t. network parameters.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got {tuple(xy.shape)}")
    xy = xy.clone().requires_grad_(True)
    value = fn(xy)
    if value.shape[-1] != 1:
        raise ValueError(f"expected a scalar field, got output {tuple(value.shape)}")

    grad = torch.autograd.grad(value.sum(), xy, create_graph=True)[0]
    lap = 0.0
    for dim in range(2):
        second = torch.autograd.grad(grad[:, dim].sum(), xy, create_graph=True)[0]
        lap = lap + second[:, dim : dim + 1]
    return value, lap


def flux_divergence(
    fn: TensorFn,
    conductivity: Callable[[torch.Tensor], torch.Tensor],
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(value, div(k grad value))`` for a scalar field, via autodiff.

    ``conductivity`` maps the field value to ``k`` and is differentiated
    through, so the ``k'(T) |grad T|^2`` term of the variable-coefficient
    operator is included automatically. Both outputs are ``(N, 1)``.
    """
    if xy.shape[-1] != 2:
        raise ValueError(f"expected (N, 2) coordinates, got {tuple(xy.shape)}")
    xy = xy.clone().requires_grad_(True)
    value = fn(xy)
    if value.shape[-1] != 1:
        raise ValueError(f"expected a scalar field, got output {tuple(value.shape)}")

    grad = torch.autograd.grad(value.sum(), xy, create_graph=True)[0]
    flux = conductivity(value) * grad                                  # (N, 2)
    div = 0.0
    for dim in range(2):
        d = torch.autograd.grad(flux[:, dim].sum(), xy, create_graph=True)[0]
        div = div + d[:, dim : dim + 1]
    return value, div


def pde_residual(
    temperature_fn: TensorFn,
    source_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xy: torch.Tensor,
    physics: Physics,
    ctx: ChipContext,
) -> torch.Tensor:
    """Strong-form residual ``-eps div(k grad theta) + theta - q_tilde``, ``(N, 1)``.

    ``temperature_fn`` returns ``theta``; ``source_fn(xy, T_celsius)`` returns
    the *total* normalised source ``q / (h_eff S)`` (it may depend on ``T``
    through the leakage term).
    """
    theta, div = flux_divergence(
        temperature_fn, lambda th: physics.k(ctx.to_celsius(th)), xy
    )
    q = source_fn(xy, ctx.to_celsius(theta))
    return -physics.eps * div + theta - q


def residual_loss(
    temperature_fn: TensorFn,
    source_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    xy: torch.Tensor,
    physics: Physics,
    ctx: ChipContext,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Mean squared PDE residual.

    The normalised residual is already O(1) (``theta`` is O(1) and
    ``eps * k`` is about 0.05), so ``residual_scale`` defaults to 1. It is kept
    as a knob for the loss-balancing ablations.
    """
    res = pde_residual(temperature_fn, source_fn, xy, physics, ctx) / residual_scale
    return torch.mean(res**2)


def data_loss(
    temperature_fn: TensorFn,
    xy: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error at the sparse sensor locations (normalised units)."""
    pred = temperature_fn(xy)
    return torch.mean((pred - target.reshape_as(pred)) ** 2)


def neumann_loss(temperature_fn: TensorFn, xy: torch.Tensor) -> torch.Tensor:
    """Adiabatic-edge penalty ``(d theta / d n)^2`` on ``dOmega``.

    Used only by the ``soft`` boundary arm. ``xy`` must lie on the edges (as
    produced by :func:`sample_boundary`): on ``x = 0`` and ``x = 1`` the normal
    derivative is ``d/dx``, on ``y = 0`` and ``y = 1`` it is ``d/dy``.
    """
    xy = xy.clone().requires_grad_(True)
    grad = torch.autograd.grad(temperature_fn(xy).sum(), xy, create_graph=True)[0]
    on_x_edge = (xy[:, 0] <= 1e-6) | (xy[:, 0] >= 1.0 - 1e-6)
    normal = torch.where(on_x_edge, grad[:, 0], grad[:, 1])
    return torch.mean(normal**2)


# --------------------------------------------------------------------------- #
# collocation / boundary sampling
# --------------------------------------------------------------------------- #
def sample_interior(n: int, generator: torch.Generator | None = None, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """``n`` uniform points in the open square ``(0, 1)^2``."""
    return torch.rand((n, 2), generator=generator, device=device, dtype=dtype)


def sample_boundary(n: int, generator: torch.Generator | None = None, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """``n`` points spread over the four edges of ``dOmega``."""
    t = torch.rand((n,), generator=generator, device=device, dtype=dtype)
    side = torch.randint(0, 4, (n,), generator=generator, device=device)
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    x = torch.where(side == 0, t, torch.where(side == 1, t, torch.where(side == 2, zeros, ones)))
    y = torch.where(side == 0, zeros, torch.where(side == 1, ones, t))
    return torch.stack([x, y], dim=1)


# --------------------------------------------------------------------------- #
# adaptive loss weighting (Wang et al., learning-rate annealing)
# --------------------------------------------------------------------------- #
def grad_norm(loss: torch.Tensor, params) -> float:
    """L2 norm of ``dloss/dparams``, ignoring parameters with no path to ``loss``."""
    params = [p for p in params if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(torch.sum(g.detach() ** 2))
    return total**0.5


def annealed_weight(reference_loss: torch.Tensor, target_loss: torch.Tensor, params, eps: float = 1e-12) -> float:
    """Weight for ``target_loss`` that equalises its gradient norm with ``reference_loss``."""
    ref = grad_norm(reference_loss, params)
    tgt = grad_norm(target_loss, params)
    if tgt < eps:
        return 1.0
    return ref / (tgt + eps)
