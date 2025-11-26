import numpy as np
from scipy.stats import bernoulli
from pathlib import Path
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import List, Optional
from argparse import ArgumentParser


def main(n: int = 100_000, true_p: float = .5) -> None:

    true_dist = bernoulli(p=true_p)

    x = true_dist.rvs(n)

    p_hat = x.mean()
    p_hat_std = np.sqrt(p_hat*(1-p_hat))


    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_p", type=float, default=.5)
    args = parser.parse_args()

    main(true_p=args.true_p)
