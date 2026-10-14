"""Model C: physics-informed DeepONet (hybrid of Model A and the PINN loss).

Same branch/trunk split as ``deeponet.py`` (Model A), with two outputs per
query point, both from the same branch embedding of the chip's sensors:

    theta(x, y) = branch_T(sensors) . trunk_T(x, y) + bias      temperature
    q_dyn(x, y) = softplus( branch_q(sensors) . trunk_q(x, y) + bias_q )  >= 0

``q_dyn`` is the (normalised, non-negative) dynamic power density. It is not
given to the model: it is predicted, and it is tied to the temperature only
through the heat-equation residual (``src/physics/losses.py``), so the extra
physics needs no extra data. Leakage is applied on top by the loss
(``Physics.leakage_factor``), exactly as in the physics-only PINN.

``hard_bc=True`` feeds the temperature trunk ``(cos pi x, cos pi y)`` instead of
``(x, y)``. Every such function has ``d theta / dn = 0`` on all four edges for
any weights (the adiabatic boundary of the dataset), by construction, the same
trick as the PINN's hard boundary.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.deeponet import _mlp


class HybridDeepONet(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        p: int = 128,
        branch_width: int = 128,
        branch_depth: int = 3,
        trunk_width: int = 128,
        trunk_depth: int = 4,
        activation: str = "tanh",
        hard_bc: bool = True,
    ):
        super().__init__()
        self.n_sensors, self.p, self.hard_bc = n_sensors, p, hard_bc
        self.branch = _mlp(3 * n_sensors, branch_width, 2 * p, branch_depth, activation)
        self.trunk_T = _mlp(2, trunk_width, p, trunk_depth, activation)
        self.trunk_q = _mlp(2, trunk_width, p, trunk_depth, activation)
        self.bias_T = nn.Parameter(torch.zeros(1))
        self.bias_q = nn.Parameter(torch.zeros(1))

    # -- embeddings ---------------------------------------------------------- #
    def embed_branch(self, branch_in: torch.Tensor) -> torch.Tensor:
        """``(batch, 3N) -> (batch, 2p)``: first p for temperature, last p for source."""
        return self.branch(branch_in)

    def _trunk_T_in(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.cos(math.pi * xy) if self.hard_bc else xy

    # -- chips x shared query points ----------------------------------------- #
    def forward(self, branch_in: torch.Tensor, xy: torch.Tensor):
        """``(batch, 3N), (Q, 2) -> theta (batch, Q), q_dyn (batch, Q)``."""
        b = self.embed_branch(branch_in)
        theta = b[:, : self.p] @ self.trunk_T(self._trunk_T_in(xy)).T + self.bias_T
        q = nn.functional.softplus(b[:, self.p:] @ self.trunk_q(xy).T + self.bias_q)
        return theta, q

    # -- one query point per row (for autograd residuals) -------------------- #
    def theta_pointwise(self, b_rep: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """``b_rep (N, 2p)`` is a chip embedding repeated per point; returns ``(N, 1)``."""
        t = self.trunk_T(self._trunk_T_in(xy))
        return (b_rep[:, : self.p] * t).sum(-1, keepdim=True) + self.bias_T

    def q_pointwise(self, b_rep: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        t = self.trunk_q(xy)
        return nn.functional.softplus((b_rep[:, self.p:] * t).sum(-1, keepdim=True) + self.bias_q)

    def predict_field(self, branch_in_single: torch.Tensor, xy_grid: torch.Tensor):
        theta, q = self.forward(branch_in_single.unsqueeze(0), xy_grid)
        return theta.squeeze(0), q.squeeze(0)


def build_hybrid(config: dict) -> HybridDeepONet:
    net = config.get("net", {})
    return HybridDeepONet(
        n_sensors=config["n_sensors"],
        p=int(net.get("p", 128)),
        branch_width=int(net.get("branch_width", 128)),
        branch_depth=int(net.get("branch_depth", 3)),
        trunk_width=int(net.get("trunk_width", 128)),
        trunk_depth=int(net.get("trunk_depth", 4)),
        activation=net.get("activation", "tanh"),
        hard_bc=bool(net.get("hard_bc", True)),
    )
