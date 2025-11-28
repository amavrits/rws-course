import numpy as np
from scipy.stats import bernoulli, norm, beta, truncnorm
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Optional, Tuple
from argparse import ArgumentParser


def plot_cis(
        prior_params: Tuple[float],
        posterior_params: Tuple[float],
        n: int,
        path: Optional[Path] = None,
        true_p: float = .5,
        alpha: float = .05,
        return_fig: bool = False
) -> plt.Figure:

    p_grid = np.linspace(1e-4, 1-1e-4, 10_000)

    prior_beta = beta(a=prior_params[0], b=prior_params[1])
    prior_p_mean = prior_beta.mean()
    prior_ci = prior_beta.ppf([0.025, 0.975])
    prior_errs = np.array([prior_p_mean-prior_ci[0], prior_ci[1]-prior_p_mean]).reshape(-1, 1)
    prior_pdf = prior_beta.pdf(p_grid)

    posterior_beta = beta(a=posterior_params[0], b=posterior_params[1])
    posterior_p_mean = posterior_beta.mean()
    posterior_ci = posterior_beta.ppf([0.025, 0.975])
    posterior_errs = np.array([posterior_p_mean-posterior_ci[0], posterior_ci[1]-posterior_p_mean]).reshape(-1, 1)
    posterior_pdf = posterior_beta.pdf(p_grid)

    fig = plt.figure(figsize=(12, 5))
    label = f"Prior mean and {(1-alpha)*100:.0f}% CI"
    plt.plot(p_grid, prior_pdf, c="b", label="Prior PDF")
    plt.errorbar(x=prior_p_mean, y=0.4*posterior_pdf.max(), xerr=prior_errs, c="b", fmt="o", capsize=5, label=label)
    label = f"Posterior mean and {(1-alpha)*100:.0f}% CI"
    plt.plot(p_grid, posterior_pdf, c="r", label="Posterior PDF")
    plt.errorbar(x=posterior_p_mean, y=0.6*posterior_pdf.max(), xerr=posterior_errs, c="r", fmt="o", capsize=5, label=label)
    plt.axvline(true_p, c="k", linestyle="--", label="True value")
    plt.xlabel("$\hat{p}$", fontsize=14)
    plt.ylabel("Density [-]", fontsize=14)
    plt.xlim(0, 1)
    plt.legend(fontsize=12, loc="upper right")
    plt.title(f"Bayesian estimates for {n} samples", fontsize=14)
    plt.grid()
    plt.close()

    if return_fig:
        return fig
    else:
        fig.savefig(path/f"bernoulli_fit_progression_{n}_obs.png")


def inference(sample: NDArray, prior_params: Tuple[float]) -> Tuple[float]:
    prior_a, prior_b = prior_params
    sample_sum = sample.sum()
    post_a = prior_a + sample_sum
    post_b = prior_b + sample.size - sample_sum
    return post_a, post_b


def main(n_all: int = 100_000, true_p: float = .5, prior_a: float = .5, prior_b: float = .5) -> None:

    prior_params = (prior_a, prior_b)

    n_obs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

    script_dir = Path(__file__).parent

    result_path = script_dir.parent / "results/bayesian_bernoulli_model"
    result_path.mkdir(parents=True, exist_ok=True)

    true_dist = bernoulli(p=true_p)

    np.random.seed(42)
    sample = true_dist.rvs(n_all)

    posterior_params = [inference(sample[:n], prior_params) for n in n_obs]

    for (n, params) in zip(n_obs, posterior_params):
        plot_cis(prior_params, params, n, result_path, true_p)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_p", type=float, default=.5)
    parser.add_argument("--prior-a", type=float, default=2.)
    parser.add_argument("--prior-b", type=float, default=2.)
    args = parser.parse_args()

    main(
        true_p=args.true_p,
        prior_a=args.prior_a,
        prior_b=args.prior_b
    )

