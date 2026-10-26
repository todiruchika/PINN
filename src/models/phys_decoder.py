"""Sensor-to-source network with an optional differentiable heat-equation solver.

A sensor encoder feeds a convolutional decoder that produces a 128x128 map. Two modes share
exactly the same network, so the only difference is whether physics is built in:

  ml      the decoder's two channels are read directly as the temperature rise theta and the heat source q_tilde
  hybrid  the decoder's second channel is the heat source q_tilde; theta is then *computed* from it by
          solving the heat equation (src.physics.grid_solve), so the temperature obeys the PDE by construction

Outputs use the dimensionless variables of the other models (theta = (T - T_amb) / 50 K).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.physics.grid_solve import solve_theta


class SensorToField(nn.Module):
    def __init__(self, n_sensors: int, mode: str = "hybrid", encoder_width: int = 256, channels: int = 64,
                 picard: int = 2, iterations: int = 12, dropout: float = 0.0) -> None:
        super().__init__()
        assert mode in ("ml", "hybrid")
        self.mode, self.picard, self.iterations, self.channels = mode, picard, iterations, channels
        self.encoder = nn.Sequential(
            nn.Linear(3 * n_sensors + 1, encoder_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(encoder_width, encoder_width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(encoder_width, channels * 8 * 8),
        )
        blocks, ch = [], channels
        for _ in range(3):                                   # 8 -> 16 -> 32 -> 64
            nxt = max(ch // 2, 16)
            blocks += [nn.ConvTranspose2d(ch, nxt, 4, 2, 1), nn.GELU(), nn.Conv2d(nxt, nxt, 3, padding=1), nn.GELU()]
            ch = nxt
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(ch, 2, 3, padding=1)

    def forward(self, features: torch.Tensor, ambient_c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(features).view(-1, self.channels, 8, 8)
        out = self.head(self.decoder(x))                     # (B, 2, 64, 64)
        out = F.interpolate(out, size=128, mode="bilinear", align_corners=True)
        if self.mode == "ml":
            return out[:, 0], out[:, 1]
        q_tilde = F.softplus(out[:, 1])                      # the source is non-negative
        theta = solve_theta(q_tilde, ambient_c, self.picard, self.iterations)
        return theta, q_tilde
