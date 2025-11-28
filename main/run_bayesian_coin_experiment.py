import numpy as np
from scipy.stats import bernoulli, norm, beta
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Optional
from argparse import ArgumentParser


def plot_cis(
        path: Optional[Path] = None,
        phats: List[float],
        n_obs: List[int],
        true_p: float = .5,
        alpha: float = .05
) -> plt.Figure:

    phats = np.array(phats)

    phat_cis = np.empty((phats.size, 2))
    for i, (n, phat) in enumerate(zip(n_obs, phats)):
        phat_std = np.sqrt(phat * (1 - phat) / n)
        phat_ci = np.clip([phat + norm.ppf(0.025) * phat_std, phat + norm.ppf(0.975) * phat_std], 0., 1.)
        phat_cis[i] = phat_ci

    phat_errs = np.array([phats-phat_cis[:, 0], phat_cis[:, 1]-phats])

    fig = plt.figure(figsize=(12, 6))
    plt.errorbar(x=n_obs, y=phats, yerr=phat_errs, c="b", fmt="o", capsize=5, label=f"Mean and {(1-alpha)*100:.0f}% CI")
    plt.axhline(true_p, c="r", linestyle="--", label="True value")
    plt.xlabel("# of observations", fontsize=14)
    plt.ylabel("$\hat{p}$", fontsize=14)
    plt.xscale("log")
    plt.ylim(-.02, 1.02)
    plt.legend(fontsize=12)
    plt.grid()
    plt.close()

    if path is not None:
        fig.savefig(path/f"bernoulli_fit_progression.png")
    else:
        return fig


def main(n_all: int = 100_000, true_p: float = .5, prior_a: float = .5, prior_b: float = .5) -> None:

    n_obs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

    script_dir = Path(__file__).parent

    data_path = script_dir.parent / "data/bernoulli_model"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_dir.parent / "results/bernoulli_model"
    result_path.mkdir(parents=True, exist_ok=True)

    true_dist = bernoulli(p=true_p)

    np.random.seed(42)
    sample = true_dist.rvs(n_all)

    post_as = [None] * len(n_obs)
    post_bs = [None] * len(n_obs)
    for i, n in enumerate(n_obs):
        x = sample[:n]
        x_sum = x.sum()
        post_a = prior_a + x_sum
        post_b = prior_b + n - x_sum
        post_as[i] = post_a
        post_ab[i] = post_b

    plot_cis(result_path, prior_a, prior_b, post_as, post_bs, n_obs, true_p)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_p", type=float, default=.5)
    parser.add_argument("--prior-a", type=float, default=.5)
    parser.add_argument("--prior-b", type=float, default=.5)
    args = parser.parse_args()

    main(
        true_p=args.true_p,
        prior_a=args.prior_a,
        prior_b=args.prior_b
    )

