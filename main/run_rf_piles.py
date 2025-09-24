import numpy as np
from scipy.stats import norm
from main.generate_rf import *
from pathlib import Path
import json
from numpy.typing import NDArray
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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


def pile_analysis(
        n_piles: int = 10,
        mean: float = 20.,
        std: float = 4.,
        theta_x: float = 100.,
        theta_y: float = 1.,
        n_x: int = 100,
        n_y: int = 50,
        load_per_pile: float = 400.,
        pile_diameter: float = 1.,
        pile_length:float = 15.,
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

    x_grid_piles = np.linspace(0, 100, n_piles+1)
    pile_xs = (x_grid_piles[:-1] + x_grid_piles[1:]) / 2
    pile_tip_ys = np.repeat(pile_length+0.5, n_piles)
    pile_tip_coords = np.c_[pile_xs, pile_tip_ys]

    d_x = pile_tip_coords[:, 0][np.newaxis, :] - coords[:, 0][:, np.newaxis]
    d_y = pile_tip_coords[:, 1][np.newaxis, :] - coords[:, 1][:, np.newaxis]
    d = np.sqrt(d_x**2+d_y**2)
    rf_tip_idx = np.argmin(d, axis=0)
    rf_tip_row_idx, rf_tip_col_idx = np.unravel_index(np.argmin(d, axis=0), rf.shape)
    rf_tip = rf[rf_tip_row_idx, rf_tip_col_idx]

    area = np.pi * pile_diameter ** 2 / 4
    resisting_force_per_pile = area * rf_tip
    resisting_force_pilegroup = np.sum(resisting_force_per_pile)
    load_pilegroup = n_piles * load_per_pile
    pile_fos = resisting_force_per_pile/load_per_pile
    fos = float(resisting_force_pilegroup/load_pilegroup)

    fig = plot_rf(
        rf=rf,
        coords=coords,
        true_mean=mean,
        true_std=std,
        pile_tip_coords=pile_tip_coords,
        pile_fos=pile_fos,
        rf_tip=rf_tip,
        n_piles=n_piles,
        fos=fos
    )

    if return_fig:
        return fos, fig
    else:
        if path is not None:
            path = path / "plots"
            path.mkdir(parents=True, exist_ok=True)
            fig.savefig(path/f"rf_foundation_plot_x_{theta_x}_y_{theta_y}_{n_piles}_piles.png")
        return fos


def plot_rf(
        rf: NDArray,
        coords: NDArray,
        pile_tip_coords: NDArray,
        pile_fos: NDArray,
        rf_tip: NDArray,
        true_mean: float = 20.,
        true_std: float = 4.,
        n_piles: int = 10,
        fos: float = 1.
) -> plt.Figure:

    su_grid = np.linspace(true_mean-5*true_std, true_mean+5*true_std, 10_000)
    true_pdf = norm(loc=true_mean, scale=true_std).pdf(su_grid)
    rf_pdf = norm(loc=rf.mean(), scale=rf.std()).pdf(su_grid)
    failure_pdf = norm(loc=rf_tip.mean(), scale=rf_tip.std()).pdf(su_grid)

    fig, axs = plt.subplots(1, 2, figsize=(22, 6), gridspec_kw={'width_ratios': [3, 1]})

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
    ax.set_title(f"Pile group FoS={fos:.2f}", fontsize=14)

    cmap = plt.get_cmap("jet")
    colornorm = Normalize(vmin=np.nanmin(pile_fos), vmax=np.nanmax(pile_fos))

    for (p_fos, pile_tip_coord) in zip(pile_fos, pile_tip_coords):
        pile_x, pile_y = pile_tip_coord
        width = 1.2
        rect = Rectangle(
            (pile_x-width/2, 0),
            width=width,
            height=pile_y,
            facecolor=cmap(colornorm(p_fos)),
            edgecolor="k",
            linewidth=1.
        )
        ax.add_patch(rect)

    sm = ScalarMappable(norm=colornorm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        location="bottom",
        fraction=0.06,
        pad=0.14
    )
    cbar.set_label("FoS per pile [-]", fontsize=14)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    ax = axs[1]
    ax.plot(su_grid, true_pdf, c="b", label="True PDF")
    ax.plot(su_grid, rf_pdf, c="r", label="PDF of RF")
    ax.plot(su_grid, failure_pdf, c="g", label="PDF of pile tips")
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
        load_per_pile: float = 400.,
        pile_diameter: float = 1.
) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parent / "data/rf_piles"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_path.parent / "results/rf_piles"
    result_path.mkdir(parents=True, exist_ok=True)

    n_piles = [1, 5, 10, 20, 40]
    n_piles_results = {}
    for n in tqdm(n_piles):
        fos = pile_analysis(
            n_piles=n,
            theta_x=theta_x,
            theta_y=theta_y,
            mean=mean,
            std=std,
            n_x=n_x,
            n_y=n_y,
            load_per_pile=load_per_pile,
            pile_diameter=pile_diameter,
            path=result_path,
            return_fig=False
        )

        n_piles_results[n] = {
            "theta_x": theta_x,
            "theta_y": theta_y,
            "fos": fos,
        }

    with open(data_path/"number_piles.json", "w") as f:
        json.dump(n_piles_results, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--theta_x", type=float, default=100.)
    parser.add_argument("--theta_y", type=float, default=1.)
    parser.add_argument("--mean", type=float, default=20.)
    parser.add_argument("--std", type=float, default=4.)
    parser.add_argument("--n_x", type=int, default=100)
    parser.add_argument("--n_y", type=int, default=50)
    parser.add_argument("--load_per_pile", type=float, default=10.)
    parser.add_argument("--pile_diameter", type=float, default=1.)
    args = parser.parse_args()

    main(
        theta_x=args.theta_x,
        theta_y=args.theta_y,
        mean=args.mean,
        std=args.std,
        n_x=args.n_x,
        n_y=args.n_y,
        load_per_pile=args.load_per_pile,
        pile_diameter=args.pile_diameter
    )


