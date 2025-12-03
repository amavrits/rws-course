import numpy as np
from scipy.stats import bernoulli, norm, beta
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Optional
from argparse import ArgumentParser


def plot_cis(
        phats: List[float],
        n_obs: List[int],
        path: Optional[Path] = None,
        true_p: float = .5,
        alpha: float = .05,
        return_fig: bool = False
) -> plt.Figure:

    phats = np.array(phats)

    phat_cis = np.empty((phats.size, 2))
    for i, (n, phat) in enumerate(zip(n_obs, phats)):
        phat_std = np.sqrt(phat * (1 - phat) / n)
        phat_ci = np.clip([phat + norm.ppf(0.025) * phat_std, phat + norm.ppf(0.975) * phat_std], 0., 1.)
        phat_cis[i] = phat_ci

    phat_errs = np.array([phats - phat_cis[:, 0], phat_cis[:, 1] - phats])

    fig = plt.figure(figsize=(12, 6))
    plt.errorbar(x=n_obs, y=phats, yerr=phat_errs, c="b", fmt="o", capsize=5,
                 label=f"Mean and {(1 - alpha) * 100:.0f}% CI")
    plt.axhline(true_p, c="r", linestyle="--", label="True value")
    plt.xlabel("# of observations", fontsize=14)
    plt.ylabel("$\hat{p}$", fontsize=14)
    plt.xscale("log")
    plt.ylim(-.02, 1.02)
    plt.legend(fontsize=12)
    plt.grid()
    plt.close()

    if return_fig:
        return fig
    else:
        fig.savefig(path / f"bernoulli_fit_progression.png", dpi=900)


def inference(sample: NDArray) -> float:
    return sample.mean()


def main(n_all: int = 100_000, true_p: float = .5) -> None:

    n_obs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]

    script_dir = Path(__file__).parent

    result_path = script_dir.parent / "results/bernoulli_model"
    result_path.mkdir(parents=True, exist_ok=True)

    true_dist = bernoulli(p=true_p)

    np.random.seed(42)
    sample = true_dist.rvs(n_all)

    phats = [inference(sample[:n]) for n in n_obs]

    plot_cis(phats, n_obs, result_path, true_p, return_fig=False)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_p", type=float, default=.5)
    args = parser.parse_args()

    main(true_p=args.true_p)

