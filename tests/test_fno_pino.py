import torch

from src.models.fno_pino import FNOPINO, build_grid_input
from src.physics.grid_residual import GRID, diffusion_term, grid_residual
from src.sensors.readers import metre_to_idx


def test_uniform_field_has_no_diffusion():
    theta = torch.full((2, GRID, GRID), 0.3)
    conductivity = torch.full((2, GRID, GRID), 80.0)
    assert torch.allclose(diffusion_term(theta, conductivity), torch.zeros_like(theta), atol=1e-6)


def test_diffusion_conserves_total_heat():
    # Adiabatic edges mean the discrete divergence telescopes to zero over the grid.
    torch.manual_seed(0)
    theta = torch.randn(2, GRID, GRID)
    conductivity = 20.0 + 100.0 * torch.rand(2, GRID, GRID)
    total = diffusion_term(theta, conductivity).sum(dim=(1, 2))
    assert torch.allclose(total, torch.zeros(2), atol=1e-2)


def test_residual_zero_for_uniform_rise_with_matching_source():
    theta = torch.full((1, GRID, GRID), 0.2)
    ambient = torch.tensor([35.0])
    # No diffusion, so R = theta - q_tilde; choosing q_tilde = theta gives zero residual.
    residual = grid_residual(theta, theta.clone(), ambient)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-6)


def test_model_output_shapes_and_backward():
    model = FNOPINO(width=8, modes=4, depth=2)
    grid_in = torch.randn(3, 5, GRID, GRID)
    theta, q_tilde = model(grid_in)
    assert theta.shape == (3, GRID, GRID)
    assert q_tilde.shape == (3, GRID, GRID)
    (theta.sum() + q_tilde.sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_sensor_readings_land_on_their_pixels():
    positions = torch.tensor([[[0.003, 0.009], [0.0105, 0.0021]]])   # (B=1, N=2, xy in metres)
    readings = torch.tensor([[40.0, 55.0]])
    ambient = torch.tensor([30.0])
    grid_in = build_grid_input(positions, readings, ambient)

    mask = grid_in[0, 1]
    values = grid_in[0, 0]
    assert mask.sum().item() == 2
    for (x, y), reading in zip(positions[0].tolist(), readings[0].tolist()):
        row, col = metre_to_idx(y), metre_to_idx(x)
        assert mask[row, col] == 1.0
        assert abs(values[row, col].item() - (reading - 30.0) / 50.0) < 1e-6
