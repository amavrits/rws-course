import numpy as np
from src.settlement.model import *
from src.settlement.utils import *
import json
from pathlib import Path
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Optional
from tqdm import tqdm


def main(true_cv: float = 2*1e-8 * (24 * 3_600), days: int = 10_000) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parents[1] / "data/settlement_analysis"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_path.parents[1] / "results/settlement_analysis"
    result_path.mkdir(parents=True, exist_ok=True)

    true_cv = np.asarray([true_cv])

    all_times = np.arange(days)

    runner = Runner(all_times=all_times, true_cv=true_cv)

    obs_times = np.arange(100, all_times.max(), 100)
    settlement_obs = sample_settlement(
        times=obs_times,
        params=runner.params,
        cv=true_cv,
        n=1
    )

    predictions = {}

    for i_obs in tqdm(range(settlement_obs.size)):

        if i_obs > 0:
            runner.bayes(s_obs=settlement_obs[:i_obs], times=obs_times[:i_obs])

        forecast_times = all_times[all_times>=obs_times[i_obs]]
        prior_prediction_mean, prior_prediction_quantiles = runner.predict(times=forecast_times, type="prior")
        posterior_prediction_mean, posterior_prediction_quantiles = runner.predict(times=forecast_times, type="posterior")

        predictions[obs_times[i_obs]] = {
            "all_times": all_times.tolist(),
            "observation_times": obs_times[:i_obs].tolist(),
            "forecast_times": forecast_times.tolist(),
            "observations": settlement_obs[:i_obs].tolist(),
            "prior_mean": prior_prediction_mean.tolist(),
            "prior_lower_quantile": prior_prediction_quantiles[0].tolist(),
            "prior_upper_quantile": prior_prediction_quantiles[1].tolist(),
            "posterior_mean": posterior_prediction_mean.tolist(),
            "posterior_lower_quantile": posterior_prediction_quantiles[0].tolist(),
            "posterior_upper_quantile": posterior_prediction_quantiles[1].tolist(),
            "cv_grid": runner.cv_grid.tolist(),
            "cv_prior_pdf": np.exp(runner.cv_grid_prior_logpdf).tolist(),
            "cv_posterior_pdf": np.exp(runner.cv_grid_logpdf).tolist()
        }

        if i_obs > 0:

            plot_predictions(
                predictions=predictions[obs_times[i_obs]],
                path=result_path,
                return_fig=False
            )

            pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_cv", type=float, default=2*1e-8 * (24 * 3_600))
    parser.add_argument("--days", type=int, default=10_000)
    args = parser.parse_args()

    main(
        true_cv=args.true_cv,
        days=args.days
    )

