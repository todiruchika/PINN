"""Grid <-> metre coordinate conversion utilities."""
import numpy as np

DIE_L = 0.012
GRID = 128


def metre_to_idx(pos_m: float, n_grid: int = GRID, L: float = DIE_L) -> int:
    """Round a metre coordinate to the nearest grid index (0-based)."""
    return int(round(pos_m / L * (n_grid - 1)))


def idx_to_metre(idx: int, n_grid: int = GRID, L: float = DIE_L) -> float:
    return idx * L / (n_grid - 1)


def sensor_mask(positions_m: list, n_grid: int = GRID, L: float = DIE_L) -> np.ndarray:
    """Return boolean mask (n_grid, n_grid) True at sensor nodes."""
    mask = np.zeros((n_grid, n_grid), dtype=bool)
    for x, y in positions_m:
        ci = metre_to_idx(x, n_grid, L)
        ri = metre_to_idx(y, n_grid, L)
        mask[ri, ci] = True
    return mask


def non_sensor_indices(positions_m: list, n_grid: int = GRID, L: float = DIE_L):
    """Return flat indices of non-sensor nodes."""
    mask = sensor_mask(positions_m, n_grid, L)
    return np.where(~mask.ravel())[0]
