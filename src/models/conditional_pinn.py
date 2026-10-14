"""One network trained across chips: sensor readings -> full temperature and source fields.

A DeepONet-style conditional field. For every chip,

* the **branch** net maps the sensor readings (and the known ambient temperature)
  to a coefficient vector, and
* the **trunk** net maps a point ``(x, y)`` to a set of basis functions;

each output is the inner product of the two, so one forward pass over a fixed
grid serves every chip at once. There are two heads sharing the trunk:

``theta(x, y) = (T - T_amb) / S``          the normalised temperature rise
``q_tilde(x, y) = q / (h_eff S) >= 0``     the normalised *total* heat source

The trunk sees ``(cos pi x, cos pi y)`` instead of ``(x, y)``. Every function of
those features has zero normal derivative on all four edges for **any** weights,
so the adiabatic boundary condition holds exactly by construction (the same
trick as the per-chip model; there is no boundary weight to tune).

Predicting the source as a second head is what lets the physics loss run without
knowing the source at test time: the PDE ties the two heads together, and the
dataset's stored ``Q`` supervises the source head during training.
"""
from __future__ import annotations

import math

import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, activation: str = "tanh"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {sorted(ACTIVATIONS)}")
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers += [nn.Linear(prev, width), ACTIVATIONS[activation]()]
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionalPINN(nn.Module):
    """Sensor readings -> ``(theta, q_tilde)`` at any set of points.

    Args:
        n_sensors: number of sensor readings (5, 8 or 10).
        latent: size of the branch/trunk inner product.
        temp_scale: fixed temperature scale ``S`` in K. ``theta = (T - T_amb) / S``.
        amb_ref, amb_span: ambient temperature is fed to the branch as
            ``(T_amb - amb_ref) / amb_span``.
        hard_neumann: use the ``cos(pi x), cos(pi y)`` trunk features (exact
            adiabatic edges). ``False`` feeds raw ``(x, y)``.
    """

    def __init__(
        self,
        n_sensors: int,
        latent: int = 64,
        branch_width: int = 128,
        branch_depth: int = 3,
        trunk_width: int = 128,
        trunk_depth: int = 4,
        activation: str = "tanh",
        temp_scale: float = 50.0,
        amb_ref: float = 35.0,
        amb_span: float = 10.0,
        hard_neumann: bool = True,
    ):
        super().__init__()
        self.cfg = dict(
            n_sensors=n_sensors, latent=latent, branch_width=branch_width, branch_depth=branch_depth,
            trunk_width=trunk_width, trunk_depth=trunk_depth, activation=activation,
            temp_scale=temp_scale, amb_ref=amb_ref, amb_span=amb_span, hard_neumann=hard_neumann,
        )
        self.n_sensors = int(n_sensors)
        self.latent = int(latent)
        self.temp_scale = float(temp_scale)
        self.amb_ref = float(amb_ref)
        self.amb_span = float(amb_span)
        self.hard_neumann = bool(hard_neumann)

        # per head: `latent` coefficients plus one bias
        self.branch = MLP(self.n_sensors + 1, 2 * (self.latent + 1), branch_width, branch_depth, activation)
        self.trunk = MLP(2, 2 * self.latent, trunk_width, trunk_depth, activation)

        # start near zero so early training is not dominated by random output scale
        with torch.no_grad():
            self.branch.net[-1].weight.mul_(0.1)
            self.branch.net[-1].bias.zero_()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: dict) -> "ConditionalPINN":
        return cls(**cfg)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ #
    def condition(self, sensors_C: torch.Tensor, T_amb_C: torch.Tensor) -> torch.Tensor:
        """Branch input ``(B, N + 1)`` from readings ``(B, N)`` and ambient ``(B,)`` in degC."""
        rise = (sensors_C - T_amb_C[:, None]) / self.temp_scale
        amb = ((T_amb_C - self.amb_ref) / self.amb_span)[:, None]
        return torch.cat([rise, amb], dim=1)

    def coefficients(self, cond: torch.Tensor) -> torch.Tensor:
        """Branch output ``(B, 2, latent + 1)``: coefficients and bias for each head."""
        return self.branch(cond).reshape(cond.shape[0], 2, self.latent + 1)

    def basis(self, xy: torch.Tensor) -> torch.Tensor:
        """Trunk output ``(..., 2, latent)`` at points ``xy`` of shape ``(..., 2)``."""
        feats = torch.cos(math.pi * xy) if self.hard_neumann else xy
        return self.trunk(feats).reshape(*xy.shape[:-1], 2, self.latent)

    # ------------------------------------------------------------------ #
    def forward(self, cond: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(theta, q_tilde)`` of shape ``(B, P)``.

        ``xy`` is ``(P, 2)`` (the same points for every chip, e.g. the grid) or
        ``(B, P, 2)`` (different points per chip, e.g. training samples).
        """
        coef = self.coefficients(cond)                                   # (B, 2, L+1)
        weights, bias = coef[..., : self.latent], coef[..., self.latent]  # (B, 2, L), (B, 2)
        phi = self.basis(xy)
        if xy.dim() == 2:
            out = torch.einsum("bhl,phl->bph", weights, phi) + bias[:, None, :]
        elif xy.dim() == 3:
            out = torch.einsum("bhl,bphl->bph", weights, phi) + bias[:, None, :]
        else:
            raise ValueError(f"xy must be (P, 2) or (B, P, 2), got {tuple(xy.shape)}")
        return out[..., 0], nn.functional.softplus(out[..., 1])

    def rows_fn(self, coef_rows: torch.Tensor):
        """Closure ``xy_flat (R, 2) -> (theta, q_tilde)`` of shape ``(R, 1)`` for the physics loss.

        ``coef_rows`` is ``(R, 2, latent + 1)``: the chip coefficients already
        repeated per point, so row ``i`` of the output depends only on row ``i``
        of ``xy_flat`` (required by :func:`src.physics.autodiff.pde_residual`).
        """
        weights, bias = coef_rows[..., : self.latent], coef_rows[..., self.latent]

        def fn(xy_flat: torch.Tensor):
            phi = self.basis(xy_flat)                                    # (R, 2, L)
            out = (weights * phi).sum(-1) + bias                         # (R, 2)
            return out[:, 0:1], nn.functional.softplus(out[:, 1:2])

        return fn
