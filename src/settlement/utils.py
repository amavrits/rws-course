import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, lognorm, rv_continuous
from scipy.integrate import trapezoid, cumulative_trapezoid
from src.settlement.model import *
from dataclasses import dataclass, field
from pathlib import Path
import json
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, List


@dataclass
class Runner:
    params: SoilParams = field(default_factory=lambda: SoilParams())
    true_cv: float = 2*1e-8 * (24 * 3_600)  # in days
    cv_mean: float = 3*1e-8 * (24 * 3_600)  # in days
    cv_std: float = 3*1e-8 * (24 * 3_600) * 0.3  # in days
    settlement_cov: float = 0.5
    cv_mu: float = field(init=False)
    cv_sigma: float = field(init=False)
    cv_prior: rv_continuous = field(init=False)
    cv_grid: NDArray = field(init=False)
    cv_grid_logpdf: NDArray = field(init=False)
    all_times: NDArray = field(default_factory=lambda: np.arange(2_000))
    settlement: NDArray = field(init=False)

    def __post_init__(self) -> None:
        self.set_cv_prior()
        self.settlement = settlement_model(times=self.all_times, params=self.params, cv=self.cv_grid)
        self.n_times = self.all_times.size

    def set_cv_prior(self) -> None:
        cov = self.cv_std / self.cv_mean
        self.cv_sigma = np.sqrt(np.log(1+cov**2))
        self.cv_mu = np.log(self.cv_mean) - 0.5 * self.cv_sigma ** 2
        self.cv_prior = lognorm(s=self.cv_sigma, scale=np.exp(self.cv_mu))
        self.cv_grid = np.linspace(self.cv_prior.ppf(0.001), self.cv_prior.ppf(0.999), 1_000)
        self.cv_grid_prior_logpdf = self.prior_logprob(self.cv_grid)
        self.cv_grid_logpdf = self.prior_logprob(self.cv_grid)

    def prior_logprob(self, x: NDArray) -> NDArray:
        return self.cv_prior.logpdf(x)

    def loglike(self, s_obs: NDArray, times: Optional[NDArray] = None) -> NDArray:
        if times is None:
            times = self.all_times
        time_idx = np.isin(self.all_times, times)
        sigma = np.sqrt(np.log(1+self.settlement_cov**2))
        mu = np.log(self.settlement[:, time_idx]) - 0.5 * sigma ** 2
        return lognorm(scale=np.exp(mu), s=sigma).logpdf(s_obs).sum(axis=-1)

    def bayes(self, s_obs: NDArray, times: Optional[NDArray] = None) -> None:
        if times is None:
            times = self.all_times
        loglike = self.loglike(s_obs=s_obs, times=times)
        logpost = self.cv_grid_logpdf + loglike
        post = np.exp(logpost)
        post /= trapezoid(post, self.cv_grid)
        self.cv_grid_logpdf = np.log(post)

    def predict(
            self,
            times: Optional[NDArray] = None,
            type: str = "posterior",
            alpha: float = 0.1
    ) -> Tuple[NDArray, NDArray]:

        if times is None:
            times = self.all_times

        time_idx = np.isin(self.all_times, times)

        if type == "prior":
            cv_logpdf = self.cv_grid_prior_logpdf.copy()
        else:
            cv_logpdf = self.cv_grid_logpdf.copy()

        cv_pdf = np.exp(cv_logpdf)

        settlement_scaled = cv_pdf[:, np.newaxis] * self.settlement
        settlement_mean = trapezoid(settlement_scaled, self.cv_grid, axis=0)

        cv_grid_cdf = cumulative_trapezoid(cv_pdf, self.cv_grid)
        idx_lower_quantile = np.argmin(np.abs(cv_grid_cdf-alpha/2))
        idx_upper_quantile = np.argmin(np.abs(cv_grid_cdf-(1-alpha/2)))
        settlement_quantiles = self.settlement[[idx_lower_quantile, idx_upper_quantile]]

        return settlement_mean[time_idx], settlement_quantiles[:, time_idx]


def plot_predictions(
        predictions: Dict[str, List[float]],
        true_cv: float,
        true_settlement: NDArray,
        path: Optional[Path] = None,
        return_fig: bool = False
) -> Optional[plt.Figure]:
    
    all_times = predictions["all_times"]
    obs_times = predictions["observation_times"]
    forecast_times = predictions["forecast_times"]
    settlement_obs = predictions["observations"]
    prior_mean = predictions["prior_mean"]
    prior_lower_quantile = predictions["prior_lower_quantile"]
    prior_upper_quantile = predictions["prior_upper_quantile"]
    posterior_mean = predictions["posterior_mean"]
    posterior_lower_quantile = predictions["posterior_lower_quantile"]
    posterior_upper_quantile = predictions["posterior_upper_quantile"]
    cv_grid = predictions["cv_grid"]
    cv_prior_pdf = predictions["cv_prior_pdf"]
    cv_posterior_pdf = predictions["cv_posterior_pdf"]

    fig, axs = plt.subplots(1, 2, figsize=(20, 6), gridspec_kw={'width_ratios': [3, 1]})

    ax = axs[0]

    if len(obs_times) > 0:
        ax.axvline(max(obs_times), c="k", linestyle="--")
        ax.scatter(obs_times, settlement_obs, color="k", marker="x", label="Observations")

    ax.plot(forecast_times, prior_mean, c="b", linewidth=1.5, label="Prior mean prediction")
    ax.fill_between(
        x=forecast_times,
        y1=prior_lower_quantile,
        y2=prior_upper_quantile,
        color="b", alpha=0.3, label="Prior 90% CI"
    )
    ax.plot(forecast_times, prior_lower_quantile, c="b", linewidth=.5)
    ax.plot(forecast_times, prior_upper_quantile, c="b", linewidth=.5)
    
    ax.plot(forecast_times, posterior_mean, c="r", linewidth=1.5, label="Posterior mean prediction")
    ax.fill_between(
        x=forecast_times,
        y1=posterior_lower_quantile,
        y2=posterior_upper_quantile,
        color="r", alpha=0.3, label="Posterior 90% CI"
    )
    ax.plot(forecast_times, posterior_lower_quantile, c="r", linewidth=.5)
    ax.plot(forecast_times, posterior_upper_quantile, c="r", linewidth=.5)

    ax.plot(all_times, true_settlement, c="g", linewidth=1.5, label="True settlement model")

    ax.set_xlabel("Time [d]", fontsize=12)
    ax.set_ylabel("Settlement [m]", fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=12)
    ax.grid()

    ax = axs[1]
    ax.plot(cv_grid, cv_prior_pdf, c="b", label="Prior PDF")
    ax.plot(cv_grid, cv_posterior_pdf, c="r", label="Posterior PDF")
    ax.axvline(true_cv, linewidth=2, c="g", label="True ${C}_{v}$")
    ax.set_xlabel("${C}_{v}$ [${m}^{2}$/d]", fontsize=12)
    ax.set_ylabel("Density [-]", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid()

    plt.close()

    if return_fig:
        return fig
    else:
        if path is not None:
            path = path / "plots"
            path.mkdir(exist_ok=True, parents=True)
            if len(obs_times) > 0:
                fig.savefig(path/f"settlement_prediction_time_{max(obs_times)}.png")
            else:
                fig.savefig(path/"settlement_prediction_time_0.png")
        return



if __name__ == "__main__":

    pass

