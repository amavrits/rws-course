import numpy as np
import pandas as pd
from scipy.stats import linregress, norm, multivariate_normal
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Dict, Tuple, Any
from tqdm import tqdm


def generate_data(
        n_obs: int = 10_000,
        true_intercept: float = 5.,
        true_slope: float = 2.,
        error_sigma: float = 1.,
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


def plot_normal_progression(
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

    fig = plt.figure(figsize=(12, 6))
    plt.axhline(true_mean, color="b", label="True mean")
    plt.axhline(true_quantiles[0], color="b", linestyle="--")
    plt.axhline(true_quantiles[1], color="b", linestyle="--", label="True 90% quantiles")
    plt.errorbar(x=n_obs, y=y_means, yerr=y_mean_errs, fmt='o-', c="r", capsize=4, label="Fitted μ and 90% CI of μ")
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
              fontsize=12,
              frameon=False)
    plt.close()
    fig.savefig(path/f"normal_fit_progression.png")


def lr_inference(
        X_all:NDArray,
        y_all: NDArray,
        true_beta: NDArray,
        error_sigma: float,
        path: Path,
        n_obs: int = 10
) -> Tuple[NDArray, NDArray]:

    path = path / "lr_model_n_observations"
    path.mkdir(exist_ok=True, parents=True)

    np.random.seed(46)
    idx = np.arange(y_all.size)
    np.random.shuffle(idx)
    y = y_all[idx[:n_obs]]
    X = X_all[idx[:n_obs]]
    X = np.vstack((np.ones_like(X), X)).T

    XTX_inv = np.linalg.inv(X.T.dot(X))
    beta = XTX_inv.dot(X.T).dot(y)
    p = X.shape[1]
    y_hat = X.dot(beta.T)

    X_spread = X_all.max() - X_all.min()
    spread_offset = .05
    X_grid_lims = (X_all.min() - X_spread * spread_offset, X_all.max() + X_spread * spread_offset)
    X_grid = np.linspace(min(X_grid_lims), max(X_grid_lims), 10_000)
    X_grid_feats = np.vstack((np.ones_like(X_grid), X_grid)).T

    y_hat_grid = X_grid_feats.dot(beta.T)
    se_mean = np.sqrt(np.sum((X_grid_feats @ XTX_inv) * X_grid_feats, axis=1)) * error_sigma
    se_pred = np.sqrt(se_mean**2+error_sigma**2)
    cov_beta = XTX_inv * error_sigma ** 2
    se_beta = np.sqrt(np.diag(cov_beta))

    model_mean = norm(loc=y_hat_grid, scale=se_mean)
    model_pred = norm(loc=y_hat_grid, scale=se_pred)
    model_beta = norm(loc=beta, scale=se_beta)

    beta_quantiles = np.vstack((model_beta.ppf(q=0.05), model_beta.ppf(q=0.95)))
    beta_error = []
    for i in range(2):
        beta_error = np.abs(np.vstack((beta[i]-beta_quantiles[0, i], beta_quantiles[1, i]-beta[i])))
    beta_error = np.array(beta_error)

    true_y_hat_grid = X_grid_feats.dot(true_beta.T)
    true_model = norm(loc=true_y_hat_grid, scale=error_sigma)
    true_quantiles = np.vstack((true_model.ppf(q=0.05), true_model.ppf(q=0.95))).T

    y_mean_quantiles = np.vstack((model_mean.ppf(q=0.05), model_mean.ppf(q=0.95))).T
    y_mean_err = np.vstack((y_hat_grid - y_mean_quantiles.min(axis=1), y_mean_quantiles.max(axis=1) - y_hat_grid))

    y_quantiles = np.vstack((model_pred.ppf(q=0.05), model_pred.ppf(q=0.95))).T
    y_err = np.vstack((y_hat_grid - y_quantiles.min(axis=1), y_quantiles.max(axis=1) - y_hat_grid))

    fig = plt.figure(figsize=(12, 6))

    plt.plot(X_grid, true_y_hat_grid, c="b", label="True model")
    plt.plot(X_grid, true_quantiles.min(1), c="b", linestyle="--", label="True 90% PI")
    plt.plot(X_grid, true_quantiles.max(1), c="b", linestyle="--")

    plt.plot(X_grid, y_hat_grid, c="r", label="Fitted model")

    plt.fill_between(x=X_grid, y1=y_mean_quantiles.min(1), y2=y_mean_quantiles.max(1), color="r", alpha=0.4, label="90% CI")
    plt.plot(X_grid, y_mean_quantiles.min(1), c="r", linewidth=.3)
    plt.plot(X_grid, y_mean_quantiles.max(1), c="r", linewidth=.3)

    plt.fill_between(x=X_grid, y1=y_quantiles.min(1), y2=y_quantiles.max(1), color="r", alpha=0.1, label="90% PI")
    plt.plot(X_grid, y_quantiles.min(1), c="r", linewidth=.3)
    plt.plot(X_grid, y_quantiles.max(1), c="r", linewidth=.3)

    plt.scatter(X[:, 1], y, color="k", marker="x", label="Data")

    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.xlim(X_all.min(), X_all.max())
    plt.ylim(y_all.min(), y_all.max())
    plt.grid()
    plt.legend(fontsize=10)
    plt.close()
    fig.savefig(path / f"lr_fit_{n_obs}_observations.png")

    pdf = multivariate_normal(mean=beta, cov=cov_beta)
    beta_grid = np.linspace(beta-2*se_beta, beta+2*se_beta, 100).T.tolist()
    beta_mesh = np.meshgrid(*beta_grid)
    beta_mesh = np.c_[[m.flatten() for m in beta_mesh]].T
    beta_pdf = pdf.pdf(beta_mesh)
    beta_pdf = beta_pdf.reshape(100, 100)

    fig = plt.figure()
    plt.contour(beta_grid[0], beta_grid[1], beta_pdf)
    plt.scatter(true_beta[0], true_beta[1], color="r", marker="x", label="True parameters", zorder=2)
    plt.xlabel("Intercept", fontsize=12)
    plt.ylabel("Slope", fontsize=12)
    plt.xlim(-4, 6)
    plt.ylim(0, 2)
    plt.grid()
    plt.close()
    fig.savefig(path/f"lr_betas_contour_{n_obs}_observations.png")

    return beta, beta_error


def plot_lr_progression(
        true_beta: NDArray,
        params: Dict[int, Tuple[NDArray, NDArray]],
        path: Path
) -> None:

    n_obs = np.array(list(params.keys()))
    betas = np.array([val[0] for val in params.values()])
    beta_errs = np.array([val[1] for val in params.values()])

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(16, 6))

    ax = axs[0]
    ax.axhline(true_beta[0], color="b", label="True intercept")
    ax.errorbar(x=n_obs, y=betas[:, 0], yerr=beta_errs[:, 0].T, fmt='o-', c="r", capsize=4, label="Fitted intercept and 90% CI of intercept")
    ax.set_ylabel("Intercept", fontsize=12)
    ax.grid()
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        fontsize=12,
        frameon=False
    )

    ax = axs[1]
    ax.axhline(true_beta[1], color="b", label="True slope")
    ax.errorbar(x=n_obs, y=betas[:, 1], yerr=beta_errs[:, 1].T, fmt='o-', c="r", capsize=4,
                label="Fitted slope and 90% CI of slope")
    ax.set_ylabel("Slope", fontsize=12)
    ax.grid()
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        fontsize=12,
        frameon=False
    )

    ax.set_xlabel("Number of observations", fontsize=12)
    ax.set_xscale("log")
    plt.close()
    fig.savefig(path/f"lr_fit_progression.png")


def main(
        true_intercept: float = 5.,
        true_slope: float = 2.,
        error_sigma: float = 1.
) -> None:

    n_obs = [5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

    script_dir = Path(__file__).parent

    data_path = script_dir.parent / "data/lr_model"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_dir.parent / "results/lr_model"
    result_path.mkdir(parents=True, exist_ok=True)

    X, y = generate_data(n_obs=100_000, true_intercept=true_intercept, true_slope=true_slope)

    data = pd.DataFrame({
        "X": X,
        "y": y,
    })
    data.to_csv(data_path/"lr_data.csv", index=False)

    normal_params = {}
    for n in tqdm(n_obs, desc="Normal model inference:"):
        params_n_obs = normal_inference(y_all=data["y"].values, path=result_path, n_obs=n)
        normal_params[n] = params_n_obs

    plot_normal_progression(y_all=data["y"].values, params=normal_params, path=result_path)

    lr_params = {}
    for n in tqdm(n_obs, desc="Linear regression inference:"):
        params_n_obs = lr_inference(
            X_all=data["X"].values,
            y_all=data["y"].values,
            true_beta=np.array([true_intercept, true_slope]),
            error_sigma=error_sigma,
            path=result_path,
            n_obs=n
        )
        lr_params[n] = params_n_obs

    plot_lr_progression(
        true_beta=np.array([true_intercept, true_slope]),
        params=lr_params,
        path=result_path
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_intercept", type=float, default=5.)
    parser.add_argument("--true_slope", type=float, default=0.2)
    parser.add_argument("--error_sigma", type=float, default=1.)
    args = parser.parse_args()

    main(
        true_intercept=args.true_intercept,
        true_slope=args.true_slope,
        error_sigma=args.error_sigma
    )

