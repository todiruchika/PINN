"""Sensor-conditioned hybrid PINN for thermal-field reconstruction."""
from __future__ import annotations

import torch
from torch import nn

from src.physics.heat import neumann_boundary_residual, pde_residual


class HybridThermalPINN(nn.Module):
    """Smooth coordinate MLP conditioned on sparse sensor observations.

    The model predicts temperature at query coordinates. Training combines a
    supervised field loss with the shared PDE residual and lateral Neumann loss.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 4,
        die_length_m: float = 0.012,
        temperature_scale_c: float = 100.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.die_length_m = die_length_m
        self.temperature_scale_c = temperature_scale_c

        self.sensor_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        layers: list[nn.Module] = []
        input_dim = hidden_dim + 3
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.Tanh()))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(
        self,
        coordinates_m: torch.Tensor,
        sensor_positions_m: torch.Tensor,
        sensor_temperatures_c: torch.Tensor,
        ambient_c: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``(batch, points)`` temperatures in Celsius."""
        batch_size, point_count, _ = coordinates_m.shape
        ambient = ambient_c.reshape(batch_size, 1)
        if sensor_mask is None:
            sensor_mask = torch.ones_like(sensor_temperatures_c, dtype=torch.bool)

        sensor_features = torch.cat(
            (
                sensor_positions_m / self.die_length_m,
                ((sensor_temperatures_c - ambient) / self.temperature_scale_c).unsqueeze(-1),
            ),
            dim=-1,
        )
        encoded_sensors = self.sensor_encoder(sensor_features)
        weights = sensor_mask.to(encoded_sensors.dtype).unsqueeze(-1)
        sensor_context = (encoded_sensors * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        query_features = coordinates_m.reshape(batch_size, point_count, 2) / self.die_length_m
        context = sensor_context.unsqueeze(1).expand(-1, point_count, -1)
        ambient_feature = ((ambient - 35.0) / 20.0).unsqueeze(1).expand(-1, point_count, -1)
        decoded = self.decoder(torch.cat((query_features, context, ambient_feature), dim=-1))
        return ambient + self.temperature_scale_c * decoded.squeeze(-1)

    def physics_residual(
        self,
        coordinates_m: torch.Tensor,
        power_density: torch.Tensor,
        sensor_positions_m: torch.Tensor,
        sensor_temperatures_c: torch.Tensor,
        ambient_c: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coordinates = coordinates_m.detach().requires_grad_(True)
        temperature = self(
            coordinates,
            sensor_positions_m,
            sensor_temperatures_c,
            ambient_c,
            sensor_mask,
        )
        return pde_residual(temperature, power_density, ambient_c, coordinates)

    def boundary_loss(
        self,
        boundary_coordinates_m: torch.Tensor,
        outward_normals: torch.Tensor,
        sensor_positions_m: torch.Tensor,
        sensor_temperatures_c: torch.Tensor,
        ambient_c: torch.Tensor,
        sensor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        coordinates = boundary_coordinates_m.detach().requires_grad_(True)
        temperature = self(
            coordinates,
            sensor_positions_m,
            sensor_temperatures_c,
            ambient_c,
            sensor_mask,
        )
        normal_gradient = neumann_boundary_residual(
            temperature, coordinates, outward_normals
        )
        return (normal_gradient * self.die_length_m / self.temperature_scale_c).square().mean()