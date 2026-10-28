"""Fourier neural operator (FNO) for the real commercial-processor thermal maps.

Input encoding (``grid_inputs``), one image per frame at the processor's native map size (i5 177x166, i7 185x154):
    0. normalised sensor reading ``(y - mu) / sd`` at the sensor pixels, 0 elsewhere
    1. sensor mask (1 at the sensor pixels)
    2. x = col / (W - 1)
    3. y = row / (H - 1)
``mu`` and ``sd`` are the training-map mean and std (``D["mu"]``, ``D["sd"]``), the same normalisation as the DeepONet.
Output: the normalised map ``(T - mu) / sd``. The maps are not periodic, so the lifted field is zero-padded before the
Fourier layers and cropped after them (the usual FNO treatment of non-periodic domains).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """Learned filter on the lowest Fourier modes of a 2D field."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.out_channels, self.modes = out_channels, modes
        scale = 1.0 / (in_channels * out_channels)
        # low frequencies along y sit at both ends of the spectrum
        self.weights_top = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))
        self.weights_bottom = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batch, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
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
        return nn.functional.gelu(self.spectral(x) + self.pointwise(x))


class RealFNO(nn.Module):
    def __init__(self, width: int = 16, modes: int = 6, depth: int = 3, hidden: int = 64, pad: int = 8,
                 in_channels: int = 4) -> None:
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(FNOBlock(width, modes) for _ in range(depth))
        self.project = nn.Sequential(nn.Conv2d(width, hidden, kernel_size=1), nn.GELU(), nn.Conv2d(hidden, 1, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        x = nn.functional.pad(self.lift(x), (0, self.pad, 0, self.pad))
        for block in self.blocks:
            x = block(x)
        return self.project(x[..., :H, :W])[:, 0]          # (B, H, W)


def grid_inputs(readings: np.ndarray, rc: np.ndarray, shape: tuple, mu: float, sd: float, n: int) -> np.ndarray:
    """``(frames, 4, H, W)`` input images from the noisy readings ``(frames, n)`` at sensor pixels ``rc[:n]``."""
    H, W = shape
    out = np.zeros((len(readings), 4, H, W), dtype=np.float32)
    out[:, 0, rc[:n, 0], rc[:n, 1]] = (readings - mu) / sd
    out[:, 1, rc[:n, 0], rc[:n, 1]] = 1.0
    out[:, 2] = np.linspace(0, 1, W)[None, :]
    out[:, 3] = np.linspace(0, 1, H)[:, None]
    return out
