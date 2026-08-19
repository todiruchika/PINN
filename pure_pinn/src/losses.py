"""PDE residual and data/boundary losses.

Shared by ``models/pinn.py`` (Model B) and, later, ``models/hybrid.py``
(Model C) so the physics is written once and validated once.

All quantities here are in *normalised* units: with
``T_norm = (T - T_boundary) / scale`` and ``f_norm = Q / (k * scale)``, the
governing equation ``-k \\nabla^2 T = Q`` becomes

    \\nabla^2 T_norm + f_norm = 0 ,      T_norm = 0 on dOmega.
"""
from __future__ import annotations

from typing import Callable

import torch

TensorFn = Callable[[torch.Tensor], torch.Tensor]


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


def pde_residual(
    temperature_fn: TensorFn,
    source_fn: TensorFn,
    xy: torch.Tensor,
) -> torch.Tensor:
    """Strong-form residual ``\\nabla^2 T_norm + f_norm`` at ``xy``, shape ``(N, 1)``."""
    _, lap = laplacian(temperature_fn, xy)
    return lap + source_fn(xy)


def residual_loss(
    temperature_fn: TensorFn,
    source_fn: TensorFn,
    xy: torch.Tensor,
    residual_scale: float = 1.0,
) -> torch.Tensor:
    """Mean squared PDE residual.

    ``residual_scale`` divides the residual before squaring. The raw residual
    lives on the scale of ``f_norm`` (order 10-100 for this dataset), so
    without it the residual term would dwarf the data term at initialisation
    purely because of units.
    """
    res = pde_residual(temperature_fn, source_fn, xy) / residual_scale
    return torch.mean(res**2)


def data_loss(
    temperature_fn: TensorFn,
    xy: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error at the sparse sensor locations (normalised units)."""
    pred = temperature_fn(xy)
    return torch.mean((pred - target.reshape_as(pred)) ** 2)


def boundary_loss(temperature_fn: TensorFn, xy: torch.Tensor, target: torch.Tensor | float = 0.0) -> torch.Tensor:
    """Penalty on ``dOmega``, used only by the soft-BC ablation arm.

    ``target`` defaults to ``0.0`` because in normalised units the Dirichlet
    value is the offset itself. Passing a tensor lets the ablation supply a
    boundary value the model was *not* told about in its construction.
    """
    pred = temperature_fn(xy)
    if not torch.is_tensor(target):
        target = torch.full_like(pred, float(target))
    return torch.mean((pred - target.reshape_as(pred)) ** 2)


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
