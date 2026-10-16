"""Extended PINN (XPINN), conditioned on the sensor readings.

The die is split into non-overlapping tiles, one small network per tile. Every point belongs to exactly
one tile, so the field can jump between tiles unless the networks are made to agree. At each shared
edge a loss penalises the mismatch of the temperature, the heat source, and the temperature gradient
across the edge (the interface conditions that define XPINN). This differs from FBPINN, where
overlapping windows blend smoothly and no interface conditions are needed.

Each tile network also receives a summary of the sensor readings from a shared encoder, exactly as in
the FBPINN model, so the two decomposition models are directly comparable.
"""
from __future__ import annotations

import torch
from torch import nn

from src.models.fbpinn import mlp


class XPINN(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        tiles_per_side: int = 3,
        embed_dim: int = 32,
        hidden: int = 32,
        interface_points: int = 16,
        fd_step: float = 0.02,
        gradient_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        self.tiles = tiles_per_side
        self.fd_step = fd_step
        self.gradient_weight = gradient_weight
        side = (torch.arange(tiles_per_side, dtype=torch.float32) + 0.5) / tiles_per_side
        cy, cx = torch.meshgrid(side, side, indexing="ij")
        self.register_buffer("centres", torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1))  # (K, 2)
        self.encoder = mlp([3 * n_sensors + 1, 64, embed_dim])
        self.nets = nn.ModuleList(mlp([2 + embed_dim, hidden, hidden, 2]) for _ in range(len(self.centres)))
        self.interfaces = self._build_interfaces(interface_points)

    def _build_interfaces(self, m: int) -> list[tuple[int, int, torch.Tensor, torch.Tensor]]:
        """(tile a, tile b, points on the shared edge, direction pointing from a into b)."""
        t, out = self.tiles, []
        along = torch.linspace(0.0, 1.0, m + 2)[1:-1]
        for iy in range(t):
            for ix in range(t):
                a = iy * t + ix
                if ix + 1 < t:   # right neighbour: vertical edge at x = (ix + 1) / t
                    y = (iy + along) / t
                    pts = torch.stack([torch.full_like(y, (ix + 1) / t), y], dim=-1)
                    out.append((a, a + 1, pts, torch.tensor([1.0, 0.0])))
                if iy + 1 < t:   # upper neighbour: horizontal edge at y = (iy + 1) / t
                    x = (ix + along) / t
                    pts = torch.stack([x, torch.full_like(x, (iy + 1) / t)], dim=-1)
                    out.append((a, a + t, pts, torch.tensor([0.0, 1.0])))
        return out

    def tile_index(self, coords: torch.Tensor) -> torch.Tensor:
        ix = torch.clamp((coords[:, 0] * self.tiles).long(), 0, self.tiles - 1)
        iy = torch.clamp((coords[:, 1] * self.tiles).long(), 0, self.tiles - 1)
        return iy * self.tiles + ix

    def _evaluate(self, k: int, points: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """Tile k's network at arbitrary points: (B, n, 2)."""
        batch, n = embedding.shape[0], points.shape[0]
        local = (points - self.centres[k]) * self.tiles                    # about [-0.5, 0.5] inside the tile
        inputs = torch.cat([local[None].expand(batch, -1, -1), embedding[:, None, :].expand(-1, n, -1)], dim=-1)
        return self.nets[k](inputs)

    def forward(self, sensor_features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sensor_features: (B, 3N + 1). coords: (P, 2). Returns theta, q_tilde, each (B, P)."""
        batch, points = sensor_features.shape[0], coords.shape[0]
        embedding = self.encoder(sensor_features)
        which = self.tile_index(coords)
        theta = torch.zeros(batch, points, device=coords.device)
        q = torch.zeros(batch, points, device=coords.device)
        for k in range(len(self.nets)):
            sel = (which == k).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            out = self._evaluate(k, coords[sel], embedding)
            theta[:, sel] = out[..., 0]
            q[:, sel] = out[..., 1]
        return theta, q

    def interface_loss(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """Mismatch of value, source, and gradient across every shared tile edge."""
        embedding = self.encoder(sensor_features)
        h = self.fd_step / self.tiles
        total = torch.zeros((), device=sensor_features.device)
        for a, b, pts, direction in self.interfaces:
            a_edge = self._evaluate(a, pts, embedding)
            b_edge = self._evaluate(b, pts, embedding)
            a_in = self._evaluate(a, pts - h * direction, embedding)      # inside tile a
            b_in = self._evaluate(b, pts + h * direction, embedding)      # inside tile b
            value = ((a_edge - b_edge) ** 2).mean()
            # Gradient on either side from a one-step difference; equal when the gradient is continuous.
            slope_a = (a_edge[..., 0] - a_in[..., 0]) / h
            slope_b = (b_in[..., 0] - b_edge[..., 0]) / h
            gradient = ((slope_a - slope_b) ** 2).mean()
            total = total + value + self.gradient_weight * gradient
        return total / len(self.interfaces)
