"""Train the sensor-conditioned hybrid thermal PINN."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from generator.pde_solver import H_EFF_DEFAULT
from src.data.dataset import RAW_DIR, load_metadata, load_sample
from src.models.hybrid_pinn import HybridThermalPINN
from src.sensors.readers import DIE_L, GRID


def _indices_to_coordinates(indices: np.ndarray) -> np.ndarray:
    rows, cols = np.divmod(indices, GRID)
    spacing = DIE_L / (GRID - 1)
    return np.stack((cols * spacing, rows * spacing), axis=-1).astype(np.float32)


def _boundary_points(count: int, rng: np.random.Generator):
    sides = rng.integers(0, 4, size=count)
    positions = rng.integers(0, GRID, size=count)
    rows = np.where(sides == 0, positions, np.where(sides == 1, positions, np.where(sides == 2, 0, GRID - 1)))
    cols = np.where(sides == 0, 0, np.where(sides == 1, GRID - 1, positions))
    normals = np.zeros((count, 2), dtype=np.float32)
    normals[sides == 0, 0] = -1.0
    normals[sides == 1, 0] = 1.0
    normals[sides == 2, 1] = -1.0
    normals[sides == 3, 1] = 1.0
    flat_indices = rows * GRID + cols
    return _indices_to_coordinates(flat_indices), normals


def _load_split(split: str, n_sensors: int, raw_dir: Path) -> list[dict]:
    metadata = load_metadata(raw_dir)
    sample_ids = metadata.loc[metadata["split"] == split, "sample_id"].tolist()
    if not sample_ids:
        raise ValueError(f"Dataset split {split!r} is empty")
    return [
        load_sample(sample_id, n_sensors, raw_dir=raw_dir, meta=metadata)
        for sample_id in sample_ids
    ]


def _tensor(value, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _training_step(
    model: HybridThermalPINN,
    optimizer: torch.optim.Optimizer,
    sample: dict,
    rng: np.random.Generator,
    device: torch.device,
    data_points: int,
    collocation_points: int,
    boundary_points: int,
    lambda_pde: float,
    lambda_boundary: float,
) -> tuple[float, float, float]:
    temperature = sample["T"]
    power = sample["Q"]
    total_pixels = temperature.size

    data_indices = rng.choice(total_pixels, data_points, replace=data_points > total_pixels)
    collocation_indices = rng.choice(
        total_pixels, collocation_points, replace=collocation_points > total_pixels
    )
    data_coordinates = _tensor(_indices_to_coordinates(data_indices), device).unsqueeze(0)
    target_temperature = _tensor(temperature.reshape(-1)[data_indices], device).unsqueeze(0)
    collocation_coordinates = _tensor(
        _indices_to_coordinates(collocation_indices), device
    ).unsqueeze(0)
    collocation_power = _tensor(power.reshape(-1)[collocation_indices], device).unsqueeze(0)

    boundary_coordinates_np, normals_np = _boundary_points(boundary_points, rng)
    boundary_coordinates = _tensor(boundary_coordinates_np, device).unsqueeze(0)
    normals = _tensor(normals_np, device).unsqueeze(0)

    sensors = sample["sensors"]
    sensor_positions = _tensor(sensors["positions"], device).unsqueeze(0)
    sensor_temperatures = _tensor(sensors["temperatures_noisy"], device).unsqueeze(0)
    ambient = _tensor([sample["meta"]["T_amb_C"]], device)

    optimizer.zero_grad(set_to_none=True)
    prediction = model(
        data_coordinates, sensor_positions, sensor_temperatures, ambient
    )
    data_mse = F.mse_loss(prediction, target_temperature)
    data_loss = data_mse / model.temperature_scale_c**2

    residual = model.physics_residual(
        collocation_coordinates,
        collocation_power,
        sensor_positions,
        sensor_temperatures,
        ambient,
    )
    residual_scale = collocation_power.abs().mean().detach() + H_EFF_DEFAULT * 50.0
    pde_loss = (residual / residual_scale).square().mean()
    boundary_loss = model.boundary_loss(
        boundary_coordinates,
        normals,
        sensor_positions,
        sensor_temperatures,
        ambient,
    )

    loss = data_loss + lambda_pde * pde_loss + lambda_boundary * boundary_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    return float(data_mse.detach()), float(pde_loss.detach()), float(boundary_loss.detach())


def _validation_mse(
    model: HybridThermalPINN,
    samples: list[dict],
    device: torch.device,
    data_points: int,
) -> float:
    model.eval()
    rng = np.random.default_rng(0)
    losses = []
    with torch.no_grad():
        for sample in samples:
            total_pixels = sample["T"].size
            indices = rng.choice(total_pixels, data_points, replace=data_points > total_pixels)
            coordinates = _tensor(_indices_to_coordinates(indices), device).unsqueeze(0)
            target = _tensor(sample["T"].reshape(-1)[indices], device).unsqueeze(0)
            sensors = sample["sensors"]
            positions = _tensor(sensors["positions"], device).unsqueeze(0)
            readings = _tensor(sensors["temperatures_noisy"], device).unsqueeze(0)
            ambient = _tensor([sample["meta"]["T_amb_C"]], device)
            losses.append(float(F.mse_loss(model(coordinates, positions, readings, ambient), target)))
    model.train()
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.validation_samples < 1:
        raise ValueError("epochs, steps_per_epoch, and validation_samples must be positive")
    train_samples = _load_split("train", args.n_sensors, args.raw_dir)
    validation_samples = _load_split("validation", args.n_sensors, args.raw_dir)
    validation_samples = validation_samples[: min(args.validation_samples, len(validation_samples))]

    model = HybridThermalPINN(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(args.steps_per_epoch):
            sample = random.choice(train_samples)
            epoch_losses.append(
                _training_step(
                    model,
                    optimizer,
                    sample,
                    np_rng,
                    device,
                    args.data_points,
                    args.collocation_points,
                    args.boundary_points,
                    args.lambda_pde,
                    args.lambda_boundary,
                )
            )

        validation_mse = _validation_mse(
            model, validation_samples, device, args.data_points
        )
        means = np.asarray(epoch_losses).mean(axis=0)
        print(
            f"epoch={epoch:04d} data_mse={means[0]:.4f} "
            f"pde={means[1]:.6f} boundary={means[2]:.6f} "
            f"validation_mse={validation_mse:.4f}"
        )
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "hidden_dim": args.hidden_dim,
                        "depth": args.depth,
                    },
                    "training_config": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "epoch": epoch,
                    "validation_mse": validation_mse,
                },
                checkpoint_path,
            )

    print(f"Best checkpoint: {checkpoint_path} (validation MSE {best_validation_mse:.4f})")
    return checkpoint_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/hybrid_pinn/best.pt"))
    parser.add_argument("--n-sensors", type=int, choices=(5, 8, 10), default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=40)
    parser.add_argument("--data-points", type=int, default=512)
    parser.add_argument("--collocation-points", type=int, default=128)
    parser.add_argument("--boundary-points", type=int, default=32)
    parser.add_argument("--validation-samples", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-pde", type=float, default=0.1)
    parser.add_argument("--lambda-boundary", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())