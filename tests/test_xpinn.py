import torch

from src.models.fbpinn import grid_coordinates
from src.models.xpinn import XPINN


def test_every_point_belongs_to_exactly_one_tile():
    model = XPINN(n_sensors=5)
    which = model.tile_index(grid_coordinates())
    assert which.min().item() == 0 and which.max().item() == 8
    counts = torch.bincount(which, minlength=9)
    assert counts.sum().item() == 128 * 128
    assert (counts > 0).all()


def test_interfaces_cover_all_neighbours():
    model = XPINN(n_sensors=5, tiles_per_side=3)
    assert len(model.interfaces) == 12     # 6 vertical edges + 6 horizontal edges in a 3x3 grid


def test_forward_shapes_and_interface_loss_backward():
    model = XPINN(n_sensors=5, embed_dim=8, hidden=8)
    features = torch.randn(2, 16)
    theta, q = model(features, grid_coordinates())
    assert theta.shape == (2, 128 * 128) and q.shape == (2, 128 * 128)
    loss = theta.sum() + q.sum() + model.interface_loss(features)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_interface_loss_is_zero_when_all_tiles_agree():
    model = XPINN(n_sensors=5, embed_dim=8, hidden=8)
    for net in model.nets:                      # make every tile network output the same constant
        for p in net.parameters():
            torch.nn.init.zeros_(p)
        net[-1].bias.data = torch.tensor([0.3, 0.7])
    loss = model.interface_loss(torch.randn(2, 16))
    assert loss.item() < 1e-10
