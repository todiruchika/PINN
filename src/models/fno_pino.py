"""Fourier neural operator (FNO) for the die temperature field, trained as a PINO.

Input: a 128x128 image built from the sparse noisy sensor readings. Each chip gives
five channels:
    0. normalised sensor reading at sensor pixels, 0 elsewhere
    1. sensor mask (1 at sensor pixels)
    2. ambient temperature, normalised, constant over the grid
    3. x coordinate in [0, 1]
    4. y coordinate in [0, 1]

Output: two 128x128 channels,
    0. theta, the temperature rise over ambient divided by S = 50 K
    1. q_tilde, the total heat source divided by h_eff * S (the source head)

The source head is needed because the heat source is not known at test time.
Only theta is scored; q_tilde is trained against the true source and used by the
physics residual (see src/physics/grid_residual.py).
"""
from __future__ import annotations

import torch
from torch import nn

from src.physics.grid_residual import GRID, TEMP_SCALE_C
from src.sensors.readers import metre_to_idx

AMBIENT_CENTRE_C = 35.0
AMBIENT_SPAN_C = 10.0


class SpectralConv2d(nn.Module):
    """Learned filter on the lowest Fourier modes of a 2D field."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Two weight blocks: low frequencies along y at both ends of the spectrum.
        self.weights_top = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )
        self.weights_bottom = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device
        )
        m = self.modes
        out_ft[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m, :m], self.weights_top)
        out_ft[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -m:, :m], self.weights_bottom)
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.spectral(x) + self.pointwise(x))


class FNOPINO(nn.Module):
    def __init__(self, width: int = 32, modes: int = 8, depth: int = 4, in_channels: int = 5) -> None:
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(depth))
        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 2, kernel_size=1),
        )

    def forward(self, grid_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.lift(grid_input)
        for block in self.blocks:
            x = block(x)
        out = self.project(x)
        return out[:, 0], out[:, 1]   # theta, q_tilde, each (B, H, W)


def build_grid_input(
    sensor_positions_m: torch.Tensor,
    sensor_values_c: torch.Tensor,
    ambient_c: torch.Tensor,
    ambient_centre_c: float = AMBIENT_CENTRE_C,
    ambient_span_c: float = AMBIENT_SPAN_C,
) -> torch.Tensor:
    """Turn sparse sensor readings into the (B, 5, H, W) input image.

    sensor_positions_m: (B, N, 2) in metres, (x, y) per sensor
    sensor_values_c:    (B, N) noisy readings in degC
    ambient_c:          (B,) ambient temperature in degC
    """
    batch, n_sensors, _ = sensor_positions_m.shape
    values = torch.zeros(batch, GRID, GRID)
    mask = torch.zeros(batch, GRID, GRID)
    positions = sensor_positions_m.detach().cpu().numpy()
    for b in range(batch):
        for s in range(n_sensors):
            x, y = positions[b, s]
            col, row = metre_to_idx(float(x)), metre_to_idx(float(y))
            values[b, row, col] = (sensor_values_c[b, s] - ambient_c[b]) / TEMP_SCALE_C
            mask[b, row, col] = 1.0

    ambient_norm = ((ambient_c - ambient_centre_c) / ambient_span_c).view(batch, 1, 1).expand(-1, GRID, GRID)
    coords = torch.linspace(0.0, 1.0, GRID)
    y_grid, x_grid = torch.meshgrid(coords, coords, indexing="ij")
    return torch.stack(
        [
            values,
            mask,
            ambient_norm,
            x_grid.expand(batch, -1, -1),
            y_grid.expand(batch, -1, -1),
        ],
        dim=1,
    )
