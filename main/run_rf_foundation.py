import numpy as np
from scipy.stats import norm
from main.generate_rf import *
from pathlib import Path
import json
from numpy.typing import NDArray
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from argparse import ArgumentParser
from tqdm import tqdm


def get_failure_points(foundation_width: float = 10., n_angles: int =50) -> Tuple[NDArray, NDArray]:
    angle_grid = np.linspace(np.pi, 0, n_angles)
    points_x = foundation_width + foundation_width * np.cos(angle_grid)
    points_y = foundation_width * np.sin(angle_grid)
    coords = np.c_[points_x, points_y]
    d_length = 2 * np.pi * foundation_width / 2 / (n_angles - 1)
    d_length = np.repeat(d_length, n_angles)
    return coords, d_length


def foundation_analysis(
        foundation_width: float = 10.,
        mean: float = 20.,
        std: float = 4.,
        theta_x: float = 100.,
        theta_y: float = 1.,
        n_x: int = 100,
        n_y: int = 50,
        foundation_load: float = 400.,
        path: Optional[Path] = None,
        return_fig: bool = False,
        random_seed: int = 42
) -> float:

    x_grid = np.linspace(0, 100, n_x)
    y_grid = np.linspace(0, 20, n_y)
    coords = np.meshgrid(x_grid, y_grid)
    coords = np.c_[[m.flatten() for m in coords]].T

    rf = make_rf(
        coords=coords,
        mean=mean,
        std=std,
        n_x=n_x,
        n_y=n_y,
        theta_x=theta_x,
        theta_y=theta_y,
        path=None,
        return_fig=False,
        random_seed=random_seed
    )

    failure_coords, d_length = get_failure_points(foundation_width=foundation_width, n_angles=100)
    failure_autocorr = markov(failure_coords, theta_x=theta_x, theta_y=theta_y)

    d_x = failure_coords[:, 0][np.newaxis, :] - coords[:, 0][:, np.newaxis]
    d_y = failure_coords[:, 1][np.newaxis, :] - coords[:, 1][:, np.newaxis]
    d = np.sqrt(d_x**2+d_y**2)
    failure_rf_idx = np.argmin(d, axis=0)
    failure_rf = rf.flatten()[failure_rf_idx]

    resisting_force = np.dot(failure_rf, d_length)
    fos = float(resisting_force/foundation_load)

    fig = plt.figure()

    fig = plot_rf(
        rf=rf,
        coords=coords,
        true_mean=mean,
        true_std=std,
        failure_coords=failure_coords,
        failure_rf=failure_rf,
        foundation_width=foundation_width,
        fos=fos
    )

    if return_fig:
        return fos, fig
    else:
        if path is not None:
            path = path / "plots"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path/f"rf_foundation_plot_x_{theta_x}_y_{theta_y}_width_{foundation_width}.png")
        return fos


def plot_rf(
        rf: NDArray,
        coords: NDArray,
        failure_coords: NDArray,
        failure_rf: NDArray,
        true_mean: float = 20.,
        true_std: float = 4.,
        foundation_width: float = 10.,
        fos: float = 1.
) -> plt.Figure:

    su_grid = np.linspace(true_mean-5*true_std, true_mean+5*true_std, 10_000)
    true_pdf = norm(loc=true_mean, scale=true_std).pdf(su_grid)
    rf_pdf = norm(loc=rf.mean(), scale=rf.std()).pdf(su_grid)
    failure_pdf = norm(loc=failure_rf.mean(), scale=failure_rf.std()).pdf(su_grid)

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [3, 1]})

    ax = axs[0]
    im = ax.imshow(rf, aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.30)
    cbar = plt.colorbar(im, cax=cax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("${S}_{u}$ [kPa]", fontsize=14, rotation=270, labelpad=20)
    ax.set_xlabel("Length [m]", fontsize=14)
    ax.set_ylabel("Depth [m]", fontsize=14)
    ax.set_ylim(coords[:, 1].min(), coords[:, 1].max())
    ax.invert_yaxis()
    ax.set_title(f"FoS={fos:.2f}", fontsize=14)

    if foundation_width >= 1.:
        ax.plot(failure_coords[:, 0], failure_coords[:, 1], c="r", linewidth=2)
        ax.plot(
            [failure_coords[:, 0].min(), failure_coords[:, 0].max()/2],
            [failure_coords[:, 1].min(), failure_coords[:, 1].min()],
            c="k",
            linewidth=10
        )
        # axin = ax.inset_axes([0.5, 0.1, 0.4, 0.4])
        # axin.scatter(failure_coords[:, 0], failure_coords[:, 1], c=failure_rf, s=20, vmin=rf.min(), vmax=rf.max())
        # ax.indicate_inset_zoom(axin, edgecolor="black")
        # axin.invert_yaxis()
        # axin.set_xticks([])
        # axin.set_yticks([])

    ax = axs[1]
    ax.plot(su_grid, true_pdf, c="b", label="True PDF")
    ax.plot(su_grid, rf_pdf, c="r", label="PDF of RF")
    ax.plot(su_grid, failure_pdf, c="g", label="PDF of failure plane")
    ax.set_xlabel("${S}_{u}$ [kPa]", fontsize=14)
    ax.set_ylabel("Density [-]", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid()

    plt.tight_layout()
    plt.close()

    return fig


def main(
        theta_x: float = 100.,
        theta_y: float = 1.,
        mean: float = 20.,
        std: float = 4.,
        n_x: int = 100,
        n_y: int = 50,
        foundation_load: float = 400.
) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parent / "data/rf_foundation"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_path.parent / "results/rf_foundation"
    result_path.mkdir(parents=True, exist_ok=True)

    foundation_widths = [1, 5, 10, 20]
    foundation_width_results = {}
    for foundation_width in tqdm(foundation_widths):
        fos = foundation_analysis(
            foundation_width=foundation_width,
            theta_x=theta_x,
            theta_y=theta_y,
            mean=mean,
            std=std,
            n_x=n_x,
            n_y=n_y,
            foundation_load=foundation_load,
            path=result_path,
            return_fig=False
        )

        foundation_width_results[foundation_width] = {
            "theta_x": theta_x,
            "theta_y": theta_y,
            "fos": fos,
        }

    with open(data_path/"foundation_width_results.json", "w") as f:
        json.dump(foundation_width_results, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--theta_x", type=float, default=100.)
    parser.add_argument("--theta_y", type=float, default=1.)
    parser.add_argument("--mean", type=float, default=20.)
    parser.add_argument("--std", type=float, default=4.)
    parser.add_argument("--n_x", type=int, default=100)
    parser.add_argument("--n_y", type=int, default=50)
    parser.add_argument("--foundation_load", type=float, default=400.)
    args = parser.parse_args()

    main(
        theta_x=args.theta_x,
        theta_y=args.theta_y,
        mean=args.mean,
        std=args.std,
        n_x=args.n_x,
        n_y=args.n_y,
        foundation_load=args.foundation_load
    )


