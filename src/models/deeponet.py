"""Model A: a plain data-driven DeepONet, no physics term anywhere.

Branch net sees one chip's sparse sensors (positions *and* readings, since
sensor placement is randomised per chip -- the network has to know *where*
each reading came from, not only its value). Trunk net sees a query
coordinate. The two embeddings are combined by a dot product plus a learned
bias, the standard DeepONet head.

This is deliberately the same trunk/branch split used across the study's
DeepONet-family models (Model A here, Model C later) so the only difference
between "no physics" and "hybrid" is which loss terms are added on top of
this network -- not the architecture.

Coordinates passed to the trunk are normalised to [0, 1]^2 (``x`` = column,
``y`` = row of the 128x128 grid).
"""
from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}


def _mlp(in_dim: int, width: int, out_dim: int, depth: int, activation: str) -> nn.Sequential:
    """``depth`` hidden layers of ``width`` units, linear output head."""
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = [nn.Linear(in_dim, width), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), act()]
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """Random Fourier positional encoding, ``xy -> [xy, sin(2*pi*B*xy), cos(2*pi*B*xy)]``.

    Plain MLPs are spectrally biased toward low frequencies, but the ground
    truth fields have sharp, rectangular hotspot edges. This is the standard
    fix (Tancik et al., 2020) and is what lets the trunk net represent an edge
    without an enormous width. ``B`` is fixed at init (not trained).
    """

    def __init__(self, n_freqs: int = 32, sigma: float = 8.0):
        super().__init__()
        self.register_buffer("B", torch.randn(2, n_freqs) * sigma)
        self.out_dim = 2 + 2 * n_freqs

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * xy @ self.B
        return torch.cat([xy, torch.sin(proj), torch.cos(proj)], dim=-1)


class DeepONetBaseline(nn.Module):
    """``T_hat(x, y) = branch(sensors) . trunk(x, y) + bias``.

    ``branch_in`` is ``(batch, 3*n_sensors)``: sensors sorted by position and
    flattened as ``[x_1, y_1, T_1, ..., x_N, y_N, T_N]`` (both positions and
    normalised temperatures).
    """

    def __init__(
        self,
        n_sensors: int,
        p: int = 128,
        branch_width: int = 128,
        branch_depth: int = 3,
        trunk_width: int = 128,
        trunk_depth: int = 4,
        activation: str = "tanh",
        fourier_freqs: int = 0,
        fourier_sigma: float = 8.0,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.p = p
        self.branch = _mlp(3 * n_sensors, branch_width, p, branch_depth, activation)
        self.fourier = FourierFeatures(fourier_freqs, fourier_sigma) if fourier_freqs > 0 else None
        trunk_in_dim = self.fourier.out_dim if self.fourier is not None else 2
        self.trunk = _mlp(trunk_in_dim, trunk_width, p, trunk_depth, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def embed_branch(self, branch_in: torch.Tensor) -> torch.Tensor:
        """``(batch, 3N) -> (batch, p)``."""
        return self.branch(branch_in)

    def embed_trunk(self, xy: torch.Tensor) -> torch.Tensor:
        """``(Q, 2) -> (Q, p)``."""
        if self.fourier is not None:
            xy = self.fourier(xy)
        return self.trunk(xy)

    def forward(self, branch_in: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Batch of chips against a *shared* set of query points.

        ``branch_in``: ``(batch, 3N)``, ``xy``: ``(Q, 2)`` -> ``(batch, Q)``.
        """
        b = self.embed_branch(branch_in)  # (batch, p)
        t = self.embed_trunk(xy)  # (Q, p)
        return b @ t.T + self.bias

    def predict_field(self, branch_in_single: torch.Tensor, xy_grid: torch.Tensor) -> torch.Tensor:
        """One chip's full field: ``branch_in_single`` ``(3N,)``, ``xy_grid`` ``(Q, 2) -> (Q,)``."""
        b = self.embed_branch(branch_in_single.unsqueeze(0))  # (1, p)
        t = self.embed_trunk(xy_grid)  # (Q, p)
        return (b @ t.T + self.bias).squeeze(0)


def build_deeponet(config: dict) -> DeepONetBaseline:
    n_sensors = config["n_sensors"]
    net_cfg = config.get("net", {})
    return DeepONetBaseline(
        n_sensors=n_sensors,
        p=int(net_cfg.get("p", 128)),
        branch_width=int(net_cfg.get("branch_width", 128)),
        branch_depth=int(net_cfg.get("branch_depth", 3)),
        trunk_width=int(net_cfg.get("trunk_width", 128)),
        trunk_depth=int(net_cfg.get("trunk_depth", 4)),
        activation=net_cfg.get("activation", "tanh"),
        fourier_freqs=int(net_cfg.get("fourier_freqs", 0)),
        fourier_sigma=float(net_cfg.get("fourier_sigma", 8.0)),
    )
