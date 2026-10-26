"""Differentiable solver for the die heat equation on the 128x128 grid.

Solves (I - c * L_k) theta = q_tilde, the same finite-volume equation as `grid_residual`
(zero flux at the edges). The conductivity k is held fixed during one solve: it is taken from a
previous temperature estimate, which is how a Picard iteration treats k(T). The operator is
symmetric positive definite, so conjugate gradients converges; the preconditioner is the exact
inverse for a constant conductivity (a cosine transform), which makes it converge in a few iterations.
Gradients flow to q_tilde through a second solve with the same operator.
"""
from __future__ import annotations

import math

import torch

from src.physics.grid_residual import (DIE_L, DIE_THICKNESS_DEFAULT, GRID, H_EFF_DEFAULT, TEMP_SCALE_C,
                                       diffusion_term, silicon_conductivity)

_C = DIE_THICKNESS_DEFAULT / (H_EFF_DEFAULT * (DIE_L / (GRID - 1)) ** 2)


def _dct_matrix(n: int) -> torch.Tensor:
    i = torch.arange(n, dtype=torch.float64)
    m = i[:, None]
    mat = math.sqrt(2.0 / n) * torch.cos(math.pi * m * (i[None, :] + 0.5) / n)
    mat[0] = mat[0] / math.sqrt(2.0)
    return mat.float()


_DCT = _dct_matrix(GRID)
_LAMBDA = 2.0 - 2.0 * torch.cos(math.pi * torch.arange(GRID, dtype=torch.float64) / GRID).float()


def _apply(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return x - _C * diffusion_term(x, k)


def _precondition(r: torch.Tensor, k_mean: torch.Tensor) -> torch.Tensor:
    dct = _DCT.to(r.device)
    lam = _LAMBDA.to(r.device)
    hat = dct @ r @ dct.T
    scale = 1.0 + _C * k_mean.view(-1, 1, 1) * (lam[:, None] + lam[None, :])
    return dct.T @ (hat / scale) @ dct


def pcg(b: torch.Tensor, k: torch.Tensor, iterations: int) -> torch.Tensor:
    k_mean = k.mean(dim=(1, 2))
    x = torch.zeros_like(b)
    r = b.clone()
    z = _precondition(r, k_mean)
    p = z.clone()
    rz = (r * z).sum((1, 2), keepdim=True)
    for _ in range(iterations):
        ap = _apply(p, k)
        alpha = rz / ((p * ap).sum((1, 2), keepdim=True) + 1e-30)
        x = x + alpha * p
        r = r - alpha * ap
        z = _precondition(r, k_mean)
        rz_new = (r * z).sum((1, 2), keepdim=True)
        p = z + rz_new / (rz + 1e-30) * p
        rz = rz_new
    return x


class _Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, iterations):
        ctx.save_for_backward(k)
        ctx.iterations = iterations
        return pcg(q, k, iterations)

    @staticmethod
    def backward(ctx, grad):
        (k,) = ctx.saved_tensors
        return pcg(grad.contiguous(), k, ctx.iterations), None, None


def solve_theta(q_tilde: torch.Tensor, ambient_c: torch.Tensor, picard: int = 2, iterations: int = 12) -> torch.Tensor:
    """Temperature rise (units of 50 K) for a heat source q_tilde, shape (B, 128, 128).

    k(T) is updated `picard` times from the previous estimate (detached); the first guess is theta = q_tilde.
    """
    theta = q_tilde.detach()
    for _ in range(picard):
        k = silicon_conductivity(ambient_c.view(-1, 1, 1) + TEMP_SCALE_C * theta).detach()
        theta = _Solve.apply(q_tilde, k, iterations)
        theta = theta.detach() if _ < picard - 1 else theta
    return theta
