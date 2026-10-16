import torch

from src.models.fbpinn import FBPINN, grid_coordinates, sensor_features


def test_window_weights_sum_to_one():
    model = FBPINN(n_sensors=5)
    weights = model.partition(grid_coordinates())
    assert weights.shape == (128 * 128, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(128 * 128), atol=1e-6)
    assert (weights >= 0).all()


def test_forward_shapes_and_backward():
    model = FBPINN(n_sensors=5, embed_dim=8, hidden=8)
    features = torch.randn(2, 3 * 5 + 1)
    theta, q = model(features, grid_coordinates())
    assert theta.shape == (2, 128 * 128)
    assert q.shape == (2, 128 * 128)
    (theta.sum() + q.sum()).backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_sensor_features_layout():
    positions = torch.tensor([[[0.006, 0.012], [0.0, 0.0]]])
    readings = torch.tensor([[40.0, 35.0]])
    ambient = torch.tensor([30.0])
    feats = sensor_features(positions, readings, ambient)
    assert feats.shape == (1, 3 * 2 + 1)
    assert abs(feats[0, 0].item() - 0.5) < 1e-6        # x = 0.006 / 0.012
    assert abs(feats[0, 2].item() - 0.2) < 1e-6        # (40 - 30) / 50
