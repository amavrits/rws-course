"""
Precompute Random Fields
========================
Generate and cache random fields for common parameter combinations.

Usage:
    python -m main.precompute_rf                    # Default parameter grid
    python -m main.precompute_rf --seeds 42 123     # Specific seeds
    python -m main.precompute_rf --quick            # Quick preset (fewer combinations)
"""

import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.generate_rf import make_rf
from src.rf_cache import save_rf_to_cache, get_coords, CACHE_DIR


def precompute_rfs(
    theta_x_values: list,
    theta_y_values: list,
    seeds: list,
    mean: float = 20.,
    std: float = 4.,
    n_x: int = 100,
    n_y: int = 50,
    cache_dir: Path = CACHE_DIR
) -> int:
    """
    Precompute random fields for all combinations of parameters.

    Returns the number of RFs generated.
    """
    coords, _, _ = get_coords(n_x, n_y)
    cache_dir.mkdir(parents=True, exist_ok=True)

    total = len(theta_x_values) * len(theta_y_values) * len(seeds)
    generated = 0

    print(f"Precomputing {total} random fields...")
    print(f"  theta_x: {theta_x_values}")
    print(f"  theta_y: {theta_y_values}")
    print(f"  seeds: {seeds}")
    print(f"  Cache dir: {cache_dir}")
    print()

    with tqdm(total=total, desc="Generating RFs") as pbar:
        for theta_x in theta_x_values:
            for theta_y in theta_y_values:
                for seed in seeds:
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
                    save_rf_to_cache(rf, theta_x, theta_y, seed, mean, std, cache_dir)
                    generated += 1
                    pbar.update(1)

    print(f"\nGenerated {generated} random fields in {cache_dir}")
    return generated


def get_default_theta_x():
    """Default theta_x values covering common use cases."""
    return [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]


def get_default_theta_y():
    """Default theta_y values covering common use cases."""
    return [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


def get_default_seeds():
    """Default seeds for reproducible random fields."""
    return [42, 123, 456, 789, 1000]


def get_quick_theta_x():
    """Quick preset: fewer theta_x values."""
    return [1.0, 10.0, 100.0, 1000.0]


def get_quick_theta_y():
    """Quick preset: fewer theta_y values."""
    return [0.1, 1.0, 10.0, 100.0]


def get_quick_seeds():
    """Quick preset: fewer seeds."""
    return [42]


def main():
    parser = ArgumentParser(description="Precompute random fields for caching")
    parser.add_argument("--theta-x", type=float, nargs="+", default=None,
                        help="Theta_x values to precompute")
    parser.add_argument("--theta-y", type=float, nargs="+", default=None,
                        help="Theta_y values to precompute")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Random seeds to precompute")
    parser.add_argument("--mean", type=float, default=20.,
                        help="Mean value for RF generation")
    parser.add_argument("--std", type=float, default=4.,
                        help="Standard deviation for RF generation")
    parser.add_argument("--quick", action="store_true",
                        help="Use quick preset (fewer combinations)")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="Cache directory path")

    args = parser.parse_args()

    if args.quick:
        theta_x_values = args.theta_x or get_quick_theta_x()
        theta_y_values = args.theta_y or get_quick_theta_y()
        seeds = args.seeds or get_quick_seeds()
    else:
        theta_x_values = args.theta_x or get_default_theta_x()
        theta_y_values = args.theta_y or get_default_theta_y()
        seeds = args.seeds or get_default_seeds()

    precompute_rfs(
        theta_x_values=theta_x_values,
        theta_y_values=theta_y_values,
        seeds=seeds,
        mean=args.mean,
        std=args.std,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
