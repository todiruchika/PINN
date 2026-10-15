import torch

from src.models.hybrid_pinn import HybridThermalPINN
from src.physics.heat import silicon_conductivity_torch


def _inputs(batch_size=2, point_count=8, sensor_count=5):
    coordinates = torch.rand(batch_size, point_count, 2) * 0.012
    sensor_positions = torch.rand(batch_size, sensor_count, 2) * 0.012
    sensor_temperatures = 30.0 + 50.0 * torch.rand(batch_size, sensor_count)
    ambient = 25.0 + 20.0 * torch.rand(batch_size)
    power_density = torch.rand(batch_size, point_count) * 1e6
    return coordinates, sensor_positions, sensor_temperatures, ambient, power_density


def test_hybrid_model_output_shape_and_finiteness():
    model = HybridThermalPINN(hidden_dim=32, depth=2)
    coordinates, positions, readings, ambient, _ = _inputs()
    prediction = model(coordinates, positions, readings, ambient)
    assert prediction.shape == coordinates.shape[:2]
    assert torch.isfinite(prediction).all()


def test_hybrid_physics_and_boundary_losses_backpropagate():
    model = HybridThermalPINN(hidden_dim=32, depth=2)
    coordinates, positions, readings, ambient, power = _inputs()
    residual = model.physics_residual(
        coordinates, power, positions, readings, ambient
    )
    normals = torch.tensor(
        [[[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]] * 2
    )
    boundary_loss = model.boundary_loss(
        coordinates[:, :4], normals, positions, readings, ambient
    )
    loss = residual.square().mean() / 1e12 + boundary_loss
    loss.backward()

    assert torch.isfinite(residual).all()
    assert torch.isfinite(boundary_loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_torch_conductivity_matches_solver_reference():
    conductivity = silicon_conductivity_torch(torch.tensor([27.0]))
    assert abs(conductivity.item() - 114.9207) < 0.001