"""Finite basis PINN (FBPINN), conditioned on the sensor readings.

The die is covered by overlapping windows, one small network per window. Each window's output is
weighted by a smooth window function, normalised so that the weights add up to one everywhere.
The blended output is a single smooth field, so no conditions are needed where windows meet
(unlike XPINN, which matches the networks at interfaces).

Each window network also receives a summary of the sensor readings, produced by a shared encoder,
so one model can reconstruct any chip from its own sensors. The layout is fixed for each sensor
count, so the sensor features are simply the flattened list of (x, y, reading) per sensor.

Coordinates are normalised to [0, 1]^2. Outputs are the same two quantities as the FNO model:
theta (temperature rise over ambient / 50 K) and q_tilde (heat source / (h_eff * 50 K)).
"""
from __future__ import annotations

import torch
from torch import nn

GRID = 128


def grid_coordinates() -> torch.Tensor:
    """(128*128, 2) coordinates in [0, 1]^2, flat index = row * 128 + col, x = col, y = row."""
    axis = torch.linspace(0.0, 1.0, GRID)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class FBPINN(nn.Module):
    def __init__(
        self,
        n_sensors: int,
        windows_per_side: int = 3,
        window_width: float = 0.35,
        embed_dim: int = 32,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        self.n_sensors = n_sensors
        side = torch.linspace(0.0, 1.0, windows_per_side)
        cy, cx = torch.meshgrid(side, side, indexing="ij")
        self.register_buffer("centres", torch.stack([cx.reshape(-1), cy.reshape(-1)], dim=-1))  # (K, 2)
        self.window_width = window_width
        self.encoder = mlp([3 * n_sensors + 1, 64, embed_dim])
        self.windows = nn.ModuleList(
            mlp([2 + embed_dim, hidden, hidden, 2]) for _ in range(len(self.centres))
        )

    def partition(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalised window weights, shape (P, K). Each row sums to one."""
        d2 = ((coords[:, None, :] - self.centres[None, :, :]) ** 2).sum(-1)
        phi = torch.exp(-d2 / (2.0 * self.window_width ** 2))
        return phi / phi.sum(dim=-1, keepdim=True)

    def forward(self, sensor_features: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """sensor_features: (B, 3N + 1). coords: (P, 2). Returns theta, q_tilde, each (B, P)."""
        batch, points = sensor_features.shape[0], coords.shape[0]
        embedding = self.encoder(sensor_features)                                 # (B, E)
        weights = self.partition(coords)                                          # (P, K)
        theta = torch.zeros(batch, points, device=coords.device)
        q = torch.zeros(batch, points, device=coords.device)
        for k, net in enumerate(self.windows):
            local = (coords - self.centres[k]) / 0.5                              # (P, 2), window-local
            inputs = torch.cat(
                [local[None].expand(batch, -1, -1), embedding[:, None, :].expand(-1, points, -1)], dim=-1
            )                                                                     # (B, P, 2 + E)
            out = net(inputs)                                                     # (B, P, 2)
            theta = theta + weights[None, :, k] * out[..., 0]
            q = q + weights[None, :, k] * out[..., 1]
        return theta, q


def sensor_features(positions_m: torch.Tensor, readings_c: torch.Tensor, ambient_c: torch.Tensor,
                    die_length_m: float = 0.012, temp_scale_c: float = 50.0) -> torch.Tensor:
    """(B, 3N + 1): normalised (x, y, rise over ambient) per sensor, then normalised ambient."""
    xy = positions_m / die_length_m
    rise = (readings_c - ambient_c[:, None]) / temp_scale_c
    per_sensor = torch.cat([xy, rise[..., None]], dim=-1).reshape(positions_m.shape[0], -1)
    ambient = ((ambient_c - 35.0) / 10.0)[:, None]
    return torch.cat([per_sensor, ambient], dim=1)
