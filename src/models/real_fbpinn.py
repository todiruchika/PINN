"""FBPINN temperature field for Ruchika's per-chip PINN (``pure_pinn/src``), used on the real thermal maps.

A finite-basis PINN splits the domain into overlapping subdomains, gives each its own small network and blends them with a
smooth partition of unity: ``theta(x, y) = sum_j w_j(c) * NN_j((c - centre_j) / half)``, ``c = (cos(pi x), cos(pi y))``.
Both the windows and the networks are functions of ``c``, so ``d theta / dn = 0`` holds exactly on all four edges, the same
hard adiabatic-edge treatment as her ``TemperatureField(bc_mode="hard")``. Layout is fixed (no search): 2 x 2 subdomains, window
softness 0.15 in ``c``, four tanh MLPs of width 32 and depth 4 (13.2k parameters in total; her single 64 x 4 MLP has 12.7k).

Only the field network changes; the source model, physics, losses, sampling and optimiser budget are hers
(``build_fbpinn`` replaces ``build_pinn`` inside ``train_pinn``).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class FBPINNField(nn.Module):
    bc_mode = "hard"

    def __init__(self, mlp_cls, parts: int = 2, softness: float = 0.15, width: int = 32, depth: int = 4,
                 activation: str = "tanh", out_scale: float = 1.0) -> None:
        super().__init__()
        self.out_scale, self.softness = float(out_scale), float(softness)
        self.half = 1.0 / parts                                           # c in [-1, 1], split into `parts` per axis
        centres = -1.0 + self.half * (2 * torch.arange(parts, dtype=torch.float32) + 1)
        cy, cx = torch.meshgrid(centres, centres, indexing="ij")
        self.register_buffer("centres", torch.stack([cx.ravel(), cy.ravel()], dim=1))   # (P, 2)
        self.register_buffer("cuts", (centres[:-1] + centres[1:]) / 2)                   # interior cut points per axis
        self.nets = nn.ModuleList(mlp_cls(in_dim=2, out_dim=1, width=width, depth=depth, activation=activation)
                                  for _ in range(parts * parts))
        self.parts = parts

    def _axis_windows(self, c: torch.Tensor) -> torch.Tensor:
        """Smooth 1D partition of unity ``(N, parts)`` from sigmoid steps at the cut points."""
        steps = torch.sigmoid((c[:, None] - self.cuts[None, :]) / self.softness)           # (N, parts - 1)
        ones, zeros = torch.ones_like(c[:, None]), torch.zeros_like(c[:, None])
        upper = torch.cat([ones, steps], dim=1)                                            # P(c > cut_{j-1})
        lower = torch.cat([steps, zeros], dim=1)                                           # P(c > cut_j)
        return upper - lower

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        c = torch.cos(math.pi * xy)
        wx, wy = self._axis_windows(c[:, 0]), self._axis_windows(c[:, 1])
        w = (wy[:, :, None] * wx[:, None, :]).reshape(len(xy), -1)                         # (N, P), rows sum to 1
        outs = torch.cat([net((c - self.centres[j]) / self.half) for j, net in enumerate(self.nets)], dim=1)
        return self.out_scale * torch.sum(w * outs, dim=1, keepdim=True)


def build_fbpinn(config: dict, physics=None):
    """Drop-in for ``pure_pinn/src/models/pinn.build_pinn`` (needs ``pure_pinn/src`` on ``sys.path``)."""
    from models.pinn import MLP, PhysicsOnlyPINN, build_source

    field_cfg = dict(config.get("field", {}))
    source_cfg = dict(config.get("source", {}))
    kind = source_cfg.pop("kind", "gaussian")
    field = FBPINNField(MLP, activation=field_cfg.get("activation", "tanh"), out_scale=field_cfg.get("out_scale", 1.0))
    return PhysicsOnlyPINN(field=field, source=build_source(kind, **source_cfg), physics=physics)
