import numpy as np
from src.settlement.model import *
from src.settlement.utils import *
import json
from pathlib import Path
from argparse import ArgumentParser
from numpy.typing import NDArray
from typing import Optional
from tqdm import tqdm


def main(
        true_cv: float = 2*1e-8 * (24 * 3_600),
        days: int = 10_000,
        settlement_cov:float = .1,
        days_interval: int = 1_000
) -> None:

    script_path = Path(__file__).parent

    data_path = script_path.parent / "data/settlement_analysis"
    data_path.mkdir(parents=True, exist_ok=True)

    result_path = script_path.parent / "results/settlement_analysis"
    result_path.mkdir(parents=True, exist_ok=True)

    true_cv = np.asarray([true_cv])

    all_times = np.arange(days)

    runner = Runner(
        all_times=all_times,
        true_cv=true_cv,
        settlement_cov=settlement_cov
    )

    true_settlement = settlement_model(times=all_times, params=runner.params, cv=true_cv)

    obs_times = np.arange(0, all_times.max(), days_interval)

    settlement_obs = sample_settlement(
        times=obs_times,
        params=runner.params,
        cv=true_cv,
        cov=runner.settlement_cov,
        n=1
    )

    sigma = np.sqrt(np.log(1 + runner.settlement_cov ** 2))
    mu = np.log(settlement_obs) - 0.5 * sigma ** 2
    observation_dist = lognorm(scale=np.exp(mu), s=sigma)
    settlement_obs_lower_quantiles = observation_dist.ppf(0.05)
    settlement_obs_upper_quantiles = observation_dist.ppf(0.95)

    predictions = {}

    for i_obs in tqdm(range(settlement_obs.size)):

        current_time = obs_times[i_obs]

        if current_time > 0:
            runner.bayes(s_obs=settlement_obs[:i_obs], times=obs_times[:i_obs])

        forecast_times = all_times[all_times>=obs_times[i_obs]]

        prior_prediction_mean, prior_prediction_quantiles = runner.predict(times=forecast_times, type="prior")
        posterior_prediction_mean, posterior_prediction_quantiles = runner.predict(times=forecast_times, type="posterior")

        prior_exceeds_target = runner.target_prob(times=forecast_times, type="prior")
        posterior_exceeds_target = runner.target_prob(times=forecast_times, type="posterior")

        predictions[int(current_time)] = {
            "all_times": all_times.tolist(),
            "observation_times": obs_times[1:i_obs+1].tolist() if current_time > 0 else [],
            "forecast_times": forecast_times.tolist(),
            "observations": settlement_obs[1:i_obs+1].tolist() if current_time > 0 else [],
            "observations_lower_quantiles": settlement_obs_lower_quantiles[1:i_obs+1].tolist() if current_time > 0 else [],
            "observations_upper_quantiles": settlement_obs_upper_quantiles[1:i_obs+1].tolist() if current_time > 0 else [],
            "prior_mean": prior_prediction_mean.tolist(),
            "prior_lower_quantile": prior_prediction_quantiles[0].tolist(),
            "prior_upper_quantile": prior_prediction_quantiles[1].tolist(),
            "posterior_mean": posterior_prediction_mean.tolist(),
            "posterior_lower_quantile": posterior_prediction_quantiles[0].tolist(),
            "posterior_upper_quantile": posterior_prediction_quantiles[1].tolist(),
            "cv_grid": runner.cv_grid.tolist(),
            "cv_prior_pdf": np.exp(runner.cv_grid_prior_logpdf).tolist(),
            "cv_posterior_pdf": np.exp(runner.cv_grid_logpdf).tolist(),
            "prior_exceeds_target": prior_exceeds_target.tolist(),
            "posterior_exceeds_target": posterior_exceeds_target.tolist()
        }

        plot_predictions(
            predictions=predictions[current_time],
            true_cv=true_cv,
            true_settlement=true_settlement,
            target_settlement=runner.target_settlement,
            path=result_path,
            return_fig=False
        )

    with open(data_path/"predictions.json", "w") as f:
        json.dump(predictions, f, indent=4)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--true_cv", type=float, default=4.*1e-8 * (24 * 3_600))
    parser.add_argument("--days", type=int, default=3_650)
    parser.add_argument("--settlement_cov", type=float, default=0.1)
    parser.add_argument("--days_interval", type=int, default=200)
    args = parser.parse_args()

    main(
        true_cv=args.true_cv,
        days=args.days,
        settlement_cov=args.settlement_cov,
        days_interval=args.days_interval
    )

