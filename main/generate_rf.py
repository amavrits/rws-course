import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from pathlib import Path
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Optional
from tqdm import tqdm


def markov(coords: NDArray, theta_x: float =1., theta_y: float = 1.) -> NDArray:

    n_points = coords.shape[0]

    coords_x = coords[:, 0]
    coords_y = coords[:, 1]

    d_x = coords_x[np.newaxis, :] - coords_x[:, np.newaxis]
    d_y = coords_y[np.newaxis, :] - coords_y[:, np.newaxis]
    d = np.sqrt(d_x**2+d_y**2)

    autocorr = np.exp(-(d_x**2/theta_x+d_y**2/theta_y))
    autocorr += np.eye(n_points) * 1e-5

    return autocorr


def make_rf(
        coords: NDArray,
        path: Optional[Path] = None,
        mean: float = 20.,
        std: float = 4.,
        theta_x: float = 1.,
        theta_y: float = 1.,
        n_x: int = 100,
        n_y: int = 50,
        return_fig: bool = False
) -> None:

    n_points = n_x * n_y
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]

    autocorr = markov(coords=coords, theta_x=theta_x, theta_y=theta_y)
    cov = autocorr * std ** 2

    np.random.seed(42)
    rf_standard = np.random.randn(n_points)
    rf = multivariate_normal(mean=np.repeat(mean, n_points), cov=cov).rvs().reshape(n_y, n_x)

    fig = plt.figure(figsize=(12, 6))
    im = plt.imshow(rf)
    cbar = plt.colorbar(im)
    plt.xlabel("Length [m]", fontsize=14)
    plt.ylabel("Depth [m]", fontsize=14)
    cbar.set_label("${S}_{u}$ [kPa]", rotation=270, labelpad=20, fontsize=14)
    plt.close()

    if return_fig:
        return fig
    else:
        if path is not None:
            path = path / "rf_plots"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path/f"rf_plot_x_{theta_x}_y_{theta_y}.png")
        return rf


def main(n_x: int = 100, n_y: int = 100, mean: float = 20., std: float = 4.) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parent / "data/rf"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_path.parent / "results/rf"
    result_path.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(0, 100, n_x)
    y_grid = np.linspace(0, 20, n_y)
    coords = np.meshgrid(x_grid, y_grid)
    coords = np.c_[[m.flatten() for m in coords]].T

    theta_x_grid = 10 ** np.arange(0, 4).astype(float)
    theta_y_grid = 10 ** np.arange(-2, 1).astype(float)
    theta_mesh = np.meshgrid(theta_x_grid, theta_y_grid)
    theta_mesh = np.c_[[m.flatten() for m in theta_mesh]].T

    rfs = []
    for (theta_x, theta_y) in tqdm(theta_mesh):
        rf = make_rf(
            coords=coords,
            mean=mean,
            std=std,
            n_x=n_x,
            n_y=n_y,
            theta_x=theta_x,
            theta_y=theta_y,
            path=result_path,
            return_fig=False
        )
        rfs.append({
            "theta_x": theta_x,
            "theta_y": theta_y,
            "x": coords[:, 0].tolist(),
            "y": coords[:, 1].tolist(),
            "rf": rf.tolist(),
        })

    with open(data_path/"rfs.json", "w") as f:
        json.dump(rfs, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_x", type=int, default=100)
    parser.add_argument("--n_y", type=int, default=50)
    parser.add_argument("--mean", type=float, default=20.)
    parser.add_argument("--std", type=float, default=4.)
    args = parser.parse_args()

    main(
        n_x=args.n_x,
        n_y=args.n_y,
        mean=args.mean,
        std=args.std
    )


