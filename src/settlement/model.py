import numpy as np
from scipy.stats import norm, lognorm
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import Optional


@dataclass
class SoilParams:
    cc: float = 0.35
    cr: float = 0.06
    cv: float = 3*1e-8 * (24 * 3_600)  # in days
    h: float = 2.
    e_o: float = 1.
    load: float = 20.
    sigma_o: float = 30.
    sigma_p: float = 20.


def calculate_final_settlement(params: SoilParams) -> float:

    sigma_final = params.sigma_o + params.load

    sf_nc1 = params.cr * np.log10(params.sigma_p/params.sigma_o)
    sf_nc2 = params.cc * np.log10(sigma_final/max(params.sigma_p, params.sigma_o))
    sf_oc = params.cr * np.log10(sigma_final/params.sigma_o)

    sf = np.where(
        sigma_final >= params.sigma_p,
        sf_nc1+sf_nc2,
        sf_oc
    )

    sf *= params.h / (1 + params.e_o)

    return sf


def consolidation(
        times: NDArray,
        params: SoilParams,
        cv: Optional[NDArray] = None,
        n_terms: int = 50
) -> NDArray:

    if cv is None:
        cv = np.asarray([params.cv])

    if len(times) < 1:
        times = [times]

    if not isinstance(times, np.ndarray):
        times = np.asarray(times)

    tv = cv[:, np.newaxis] * times / params.h ** 2

    summation_terms = np.arange(1, n_terms+1)[:, np.newaxis, np.newaxis]
    series_terms = np.exp(-(2*summation_terms-1)**2*np.pi**2*tv/4)
    series_terms *= 1 / (2 * summation_terms -1) **2

    doc = 1 - 8 / np.pi ** 2 * series_terms.sum(axis=0)

    return doc.squeeze()


def stress_model(doc: NDArray, params: SoilParams) -> NDArray:
    return params.sigma_o + doc * params.load


def settlement_model(
        times: float | NDArray,
        params: SoilParams,
        cv: Optional[NDArray] = None
) -> NDArray:

    sf = calculate_final_settlement(params)

    doc = consolidation(times, params, cv)

    return doc*sf


def sample_settlement(
        times: NDArray,
        params: SoilParams,
        cv: NDArray,
        n: int = 1,
        cov: float = .5,
        seed: int = 42
) -> NDArray:

    s = settlement_model(times, params, cv)

    sigma = np.sqrt(np.log(1+cov**2))
    mu = np.log(s) - 0.5 * sigma ** 2

    np.random.seed(seed)
    sample = lognorm(scale=np.exp(mu), s=sigma).rvs((n, times.size))

    return sample.squeeze()


if __name__ == "__main__":

    pass

