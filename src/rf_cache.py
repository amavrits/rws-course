"""
Random Field Cache Module
=========================
Provides fast loading of precomputed random fields with on-the-fly fallback.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from numpy.typing import NDArray
import hashlib


# Default cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "rf_cache"

# Grid configuration (must match precomputation)
N_X = 100
N_Y = 50
X_MAX = 100
Y_MAX = 20


def get_cache_key(theta_x: float, theta_y: float, seed: int, mean: float = 20., std: float = 4.) -> str:
    """Generate a unique cache key for the given parameters."""
    # Round to avoid floating point issues
    key = f"tx{theta_x:.6f}_ty{theta_y:.6f}_s{seed}_m{mean:.2f}_std{std:.2f}"
    return key


def get_cache_path(theta_x: float, theta_y: float, seed: int,
                   mean: float = 20., std: float = 4.,
                   cache_dir: Optional[Path] = None) -> Path:
    """Get the path to the cached RF file."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    key = get_cache_key(theta_x, theta_y, seed, mean, std)
    return cache_dir / f"{key}.npz"


def load_cached_rf(theta_x: float, theta_y: float, seed: int,
                   mean: float = 20., std: float = 4.,
                   cache_dir: Optional[Path] = None) -> Optional[NDArray]:
    """
    Load a precomputed random field from cache.

    Returns None if the RF is not cached.
    """
    path = get_cache_path(theta_x, theta_y, seed, mean, std, cache_dir)
    if path.exists():
        data = np.load(path)
        return data["rf"]
    return None


def save_rf_to_cache(rf: NDArray, theta_x: float, theta_y: float, seed: int,
                     mean: float = 20., std: float = 4.,
                     cache_dir: Optional[Path] = None) -> Path:
    """Save a random field to cache."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = get_cache_path(theta_x, theta_y, seed, mean, std, cache_dir)
    np.savez_compressed(path, rf=rf, theta_x=theta_x, theta_y=theta_y,
                        seed=seed, mean=mean, std=std)
    return path


def get_coords(n_x: int = N_X, n_y: int = N_Y,
               x_max: float = X_MAX, y_max: float = Y_MAX) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Get the coordinate grid for random field generation.

    Returns:
        coords: (n_points, 2) array of coordinates
        x_grid: (n_x,) array of x values
        y_grid: (n_y,) array of y values
    """
    x_grid = np.linspace(0, x_max, n_x)
    y_grid = np.linspace(0, y_max, n_y)
    mesh = np.meshgrid(x_grid, y_grid)
    coords = np.c_[[m.flatten() for m in mesh]].T
    return coords, x_grid, y_grid


def list_cached_rfs(cache_dir: Optional[Path] = None) -> list:
    """List all cached random fields."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    if not cache_dir.exists():
        return []
    return list(cache_dir.glob("*.npz"))


def get_or_generate_rf(theta_x: float, theta_y: float, seed: int,
                       mean: float = 20., std: float = 4.,
                       coords: Optional[NDArray] = None,
                       n_x: int = N_X, n_y: int = N_Y,
                       cache_dir: Optional[Path] = None,
                       save_if_generated: bool = False) -> NDArray:
    """
    Get a random field from cache, or generate it if not cached.

    This is the main function to use in Dash callbacks for fast RF loading.

    Args:
        theta_x: Horizontal correlation length
        theta_y: Vertical correlation length
        seed: Random seed
        mean: Mean value of the RF
        std: Standard deviation of the RF
        coords: Coordinate grid (generated if None)
        n_x: Number of x grid points
        n_y: Number of y grid points
        cache_dir: Cache directory path
        save_if_generated: Whether to save newly generated RFs to cache

    Returns:
        Random field array of shape (n_y, n_x)
    """
    # Try loading from cache first
    rf = load_cached_rf(theta_x, theta_y, seed, mean, std, cache_dir)
    if rf is not None:
        return rf

    # Generate if not cached
    if coords is None:
        coords, _, _ = get_coords(n_x, n_y)

    # Import here to avoid circular imports
    from main.generate_rf import make_rf

    rf = make_rf(
        coords=coords,
        mean=mean,
        std=std,
        theta_x=theta_x,
        theta_y=theta_y,
        n_x=n_x,
        n_y=n_y,
        return_fig=False,
        random_seed=seed
    )

    if save_if_generated:
        save_rf_to_cache(rf, theta_x, theta_y, seed, mean, std, cache_dir)

    return rf
