from tokenize import group

import numpy as np
import pandas as pd
from scipy.stats import linregress, norm
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Any


def generate_data(
        n_obs: int = 10_000,
        true_intercept: float = 5.,
        true_slope: float = 2.,
        error_sigma: float = 1.,
        n_groups: int = 5
) -> Tuple[NDArray, NDArray]:

    np.random.seed(42)
    X = norm(loc=10., scale=2.).rvs(n_obs)

    y_hat = true_intercept + true_slope * X

    np.random.seed(43)
    y = norm(loc=y_hat, scale=error_sigma).rvs(n_obs)

    return X, y


def normal_inference(y_all: NDArray, path: Path, n_obs: int = 10) -> Tuple[float, NDArray, NDArray]:

    path = path / "normal_model_n_observations"
    path.mkdir(exist_ok=True, parents=True)

    true_mean = y_all.mean()
    true_std = y_all.std()
    true_pdf = norm(loc=true_mean, scale=true_std)

    np.random.seed(45)
    np.random.shuffle(y_all.copy())
    y = y_all[:n_obs]
    
    y_mean = y.mean()
    pdf = norm(loc=y_mean, scale=true_std)
    mean_pdf = norm(loc=y_mean, scale=true_std/np.sqrt(n_obs))
    
    y_spread = y_all.max() - y_all.min()
    spread_offset = 1.
    grid_lims = (y_all.min()-y_spread*spread_offset, y_all.max()+y_spread*spread_offset)
    y_grid = np.linspace(min(grid_lims), max(grid_lims), 10_000)
    true_pdf = true_pdf.pdf(y_grid)
    y_pdf = pdf.pdf(y_grid)
    y_quantiles = pdf.ppf(q=[0.05, 0.95])
    y_err = np.vstack((y_mean - y_quantiles.min(), y_quantiles.max()-y_mean))
    y_mean_quantiles = mean_pdf.ppf(q=[0.05, 0.95])
    y_mean_err = np.vstack((y_mean - y_mean_quantiles.min(), y_mean_quantiles.max()-y_mean))

    fig = plt.figure()
    plt.plot(y_grid, true_pdf, c="b", label="True distribution")
    plt.axvline(true_mean, c="b", linestyle="--", label="True μ")
    plt.plot(y_grid, y_pdf, c="r", label="Fitted distribution")
    for obs in y:
        plt.axvline(obs, ymin=0, ymax=0.05, c="k")
    plt.axvline(obs, ymin=0, ymax=0.05, c="k", label="Observations")
    plt.fill_between(x=y_grid, y1=np.zeros_like(y_grid), y2=y_pdf, color="r", alpha=0.3)
    plt.errorbar(y=0.6*y_pdf.max(), x=y_mean, xerr=y_err, fmt='o-', capsize=4, c="r", label="90% CI of μ")
    plt.xlabel("Y", fontsize=12)
    plt.ylabel("Probability density [-]", fontsize=12)
    plt.xlim(y_grid.min(), y_grid.max())
    plt.ylim(0, max(y_pdf.max(), true_pdf.max())*1.1)
    plt.grid()
    plt.legend(fontsize=10)
    plt.close()
    fig.savefig(path/f"normal_fit_{n_obs}_observations.png")    

    return float(y_mean), y_mean_err.squeeze(), y_err.squeeze()


def plot_progression(
        y_all: NDArray,
        params: Dict[int, Tuple[float, NDArray, NDArray]],
        path: Path
) -> None:

    true_mean = y_all.mean()
    true_std = y_all.std()
    true_pdf = norm(loc=true_mean, scale=true_std)
    true_quantiles = true_pdf.ppf(q=[0.05, 0.95])

    n_obs = np.array(list(params.keys()))
    y_means = np.array([val[0] for val in params.values()])
    y_mean_errs = np.array([val[1] for val in params.values()]).T
    y_errs = np.array([val[2] for val in params.values()]).T

    fig = plt.figure()
    plt.axhline(true_mean, color="b", label="True mean")
    plt.axhline(true_quantiles[0], color="b", linestyle="--")
    plt.axhline(true_quantiles[1], color="b", linestyle="--", label="True 90% quantiles")
    plt.errorbar(x=n_obs, y=y_means, yerr=y_mean_errs, fmt='o-', c="r", capsize=4, label="Fitted μ and 90% quantiles of μ")
    line, caplines, barlinecols = plt.errorbar(x=n_obs, y=y_means, yerr=y_errs, fmt='--', c="r", capsize=4, label="90% quantiles")
    for blc in barlinecols:
        blc.set_linestyle('--')
    plt.xlabel("Number of observations", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.xscale("log")
    plt.grid()
    plt.legend(loc='upper center',
              bbox_to_anchor=(0.5, 1.15),
              ncol=2,
              fontsize=10,
              frameon=False)
    plt.close()
    fig.savefig(path/f"normal_fit_progression.png")


def main(
        n_obs: int = 10,
        true_intercept: float = 5.,
        true_slope: float = 2.,
        error_sigma: float = 1.
) -> None:

    script_dir = Path(__file__).parent

    data_path = script_dir.parent / "data/lr_model"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_dir.parent / "results/lr_model"
    result_path.mkdir(parents=True, exist_ok=True)

    X, y = generate_data(n_obs=100_000)

    data = pd.DataFrame({
        "X": X,
        "y": y,
    })
    data.to_csv(data_path/"lr_data.csv", index=False)

    params = {}
    for n in [5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]:
        params_n_obs = normal_inference(data["y"].values, result_path, n)
        params[n] = params_n_obs

    plot_progression(data["y"].values, params, result_path)


    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--n_obs", type=int, default=10)
    parser.add_argument("--true_intercept", type=float, default=5.)
    parser.add_argument("--true_slope", type=float, default=0.2)
    parser.add_argument("--error_sigma", type=float, default=1.)
    args = parser.parse_args()

    main(
        n_obs=args.n_obs,
        true_intercept=args.true_intercept,
        true_slope=args.true_slope,
        error_sigma=args.error_sigma
    )

