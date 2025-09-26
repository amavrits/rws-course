import numpy as np
from scipy.stats import norm
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Optional
from argparse import ArgumentParser


def main(n: int = 100_000, true_mean: float = 20., true_std: float = 4.) -> None:

    true_dist = norm(loc=true_mean, scale=true_std)

    np.random.seed(42)
    x = true_dist.rvs(n)

    mean = x.mean()
    std = x.std()
    mean_std = std / np.sqrt(n)

    fit_dist = norm(loc=mean, scale=std)
    param_dist = norm(loc=mean, scale=mean_std)

    x_grid = np.linspace(true_dist.ppf(0.001), true_dist.ppf(0.999), 10_000)
    true_pdf = true_dist.pdf(x_grid)
    fit_pdf = fit_dist.pdf(x_grid)
    param_pdf = param_dist.pdf(x_grid)




if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_mean", type=float, default=20.)
    parser.add_argument("--true_std", type=float, default=4.)
    args = parser.parse_args()

    main(
        true_mean=args.true_mean,
        true_std=args.true_std
    )