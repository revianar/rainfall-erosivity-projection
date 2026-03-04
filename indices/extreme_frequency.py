"""
extreme_frequency.py
--------------------
Extreme Rainfall Frequency Analysis for the Jakarta region.

Methods:
    1. Annual Maxima Series (AMS) extraction
    2. Distribution fitting: GEV and Gumbel
    3. Parameter estimation: L-moments (preferred) and MLE
    4. Return level computation: 2, 5, 10, 25, 50, 100 years
    5. Confidence intervals via bootstrap
    6. Change in return levels across RCP scenarios
    7. Flood-relevant threshold exceedance probabilities

GEV distribution:
    G(x) = exp{-[1 + ξ(x-μ)/σ]^(-1/ξ)}
    ξ = 0: Gumbel (Type I), ξ > 0: Fréchet (Type II), ξ < 0: Weibull (Type III)

Reference:
    Coles (2001). An Introduction to Statistical Modeling of Extreme Values.
    Hosking & Wallis (1997). Regional Frequency Analysis.
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import logging
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Return periods to compute [years]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

# Jakarta flood-relevant thresholds [mm/day]
JAKARTA_FLOOD_THRESHOLDS = {
    "moderate_flood": 100.0,  # mm/day — widespread moderate flooding
    "severe_flood": 150.0,    # mm/day — severe inundation events
    "extreme_flood": 200.0,   # mm/day — catastrophic events (e.g., 2007, 2020)
}


# ── L-moments ─────────────────────────────────────────────────────────────────

def lmoments(x: np.ndarray) -> tuple:
    """
    Compute the first four L-moments of sample x.

    Returns: (l1, l2, l3, l4) where:
        l1 = mean (location)
        l2 = scale (L-scale)
        l3, l4 used to compute L-skewness and L-kurtosis
    """
    n = len(x)
    x_sorted = np.sort(x)

    # Probability weighted moments
    b = np.zeros(4)
    for r in range(4):
        weights = np.array([
            np.prod([(j - k) for k in range(r)]) / np.prod([(n - 1 - k) for k in range(r)])
            for j in range(n)
        ])
        b[r] = np.mean(x_sorted * weights)

    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return l1, l2, l3, l4


# ── GEV Fitting ───────────────────────────────────────────────────────────────

def fit_gev_lmoments(ams: np.ndarray) -> dict:
    """
    Fit GEV distribution using L-moments estimation.

    Parameters
    ----------
    ams : np.ndarray
        Annual maximum series (sorted ascending)

    Returns
    -------
    dict with keys: loc (μ), scale (σ), shape (ξ), method
    """
    l1, l2, l3, l4 = lmoments(ams)
    t3 = l3 / l2  # L-skewness

    # Hosking (1985) approximation for shape parameter
    if abs(t3) < 1e-6:
        shape = 0.0
    else:
        c = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
        shape = 7.8590 * c + 2.9554 * c ** 2

    if abs(shape) < 1e-6:  # Gumbel limit
        scale = l2 / np.log(2)
        loc = l1 - 0.5772 * scale
    else:
        gamma_val = stats.gamma.pdf(1 + shape, 1 + shape)  # Gamma(1 + ξ)
        scale = shape * l2 / ((1 - 2 ** (-shape)) * stats.gamma(1 + shape))
        loc = l1 - scale * (1 - stats.gamma(1 + shape)) / shape

    return {"loc": loc, "scale": scale, "shape": shape, "method": "L-moments"}


def fit_gev_mle(ams: np.ndarray) -> dict:
    """
    Fit GEV distribution using Maximum Likelihood Estimation.

    Parameters
    ----------
    ams : np.ndarray
        Annual maximum series

    Returns
    -------
    dict with keys: loc, scale, shape, method, nll (neg log-likelihood)
    """
    shape, loc, scale = stats.genextreme.fit(ams)
    nll = -np.sum(stats.genextreme.logpdf(ams, shape, loc=loc, scale=scale))
    return {"loc": loc, "scale": scale, "shape": -shape, "method": "MLE", "nll": nll}
    # Note: scipy uses -ξ convention for shape


def compute_return_levels(
    loc: float, scale: float, shape: float,
    return_periods: list = RETURN_PERIODS,
) -> dict:
    """
    Compute GEV return levels for given return periods.

    x_T = μ - (σ/ξ) × [1 - (-log(1-1/T))^(-ξ)]

    Parameters
    ----------
    loc, scale, shape : float
        GEV parameters (μ, σ, ξ)
    return_periods : list
        Return periods in years

    Returns
    -------
    dict {T: return_level} in mm/day
    """
    return_levels = {}
    for T in return_periods:
        p = 1 - 1.0 / T  # non-exceedance probability
        yT = -np.log(p)   # reduced variate

        if abs(shape) < 1e-6:  # Gumbel
            xT = loc - scale * np.log(yT)
        else:
            xT = loc - (scale / shape) * (1 - yT ** (-shape))

        return_levels[T] = xT
    return return_levels


def bootstrap_return_levels(
    ams: np.ndarray,
    return_periods: list = RETURN_PERIODS,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    method: str = "lmoments",
) -> dict:
    """
    Bootstrap confidence intervals for return levels.

    Returns
    -------
    dict {T: (lower_CI, median, upper_CI)}
    """
    n = len(ams)
    rl_samples = {T: [] for T in return_periods}

    for _ in range(n_bootstrap):
        sample = np.random.choice(ams, size=n, replace=True)
        try:
            if method == "lmoments":
                params = fit_gev_lmoments(sample)
            else:
                params = fit_gev_mle(sample)
            rls = compute_return_levels(params["loc"], params["scale"], params["shape"],
                                        return_periods)
            for T, rl in rls.items():
                rl_samples[T].append(rl)
        except Exception:
            pass

    alpha = (1 - ci) / 2
    ci_results = {}
    for T in return_periods:
        arr = np.array(rl_samples[T])
        arr = arr[~np.isnan(arr)]
        ci_results[T] = (
            np.percentile(arr, alpha * 100),
            np.percentile(arr, 50),
            np.percentile(arr, (1 - alpha) * 100),
        )
    return ci_results


# ── Threshold Exceedance ───────────────────────────────────────────────────────

def exceedance_probability(
    loc: float, scale: float, shape: float,
    threshold: float,
) -> float:
    """
    Probability of exceedance P(X > threshold) using fitted GEV.
    Returns annual exceedance probability (= 1/return_period).
    """
    if abs(shape) < 1e-6:
        p_exceed = np.exp(-np.exp(-(threshold - loc) / scale))
    else:
        z = 1 + shape * (threshold - loc) / scale
        if z <= 0:
            return 1.0
        p_exceed = np.exp(-(z ** (-1 / shape)))
    return 1 - p_exceed  # exceedance prob


# ── Spatial Frequency Analysis ────────────────────────────────────────────────

def spatial_frequency_analysis(
    ds_indices: xr.Dataset,
    variable: str = "Rx1day",
    return_periods: list = RETURN_PERIODS,
    n_bootstrap: int = 500,
) -> xr.Dataset:
    """
    Apply GEV frequency analysis at each grid cell.

    Parameters
    ----------
    ds_indices : xr.Dataset
        Indices dataset with dims (year, lat, lon)
    variable : str
        Which variable to analyse (Rx1day, Rx3day, Rx5day)

    Returns
    -------
    xr.Dataset
        Return levels, GEV parameters, and exceedance probs for each grid cell
    """
    da = ds_indices[variable]
    nlat = len(ds_indices.lat)
    nlon = len(ds_indices.lon)

    # Output arrays
    rl_data = {T: np.full((nlat, nlon), np.nan) for T in return_periods}
    rl_lower = {T: np.full((nlat, nlon), np.nan) for T in return_periods}
    rl_upper = {T: np.full((nlat, nlon), np.nan) for T in return_periods}
    gev_loc = np.full((nlat, nlon), np.nan)
    gev_scale = np.full((nlat, nlon), np.nan)
    gev_shape = np.full((nlat, nlon), np.nan)
    exceedance = {k: np.full((nlat, nlon), np.nan) for k in JAKARTA_FLOOD_THRESHOLDS}

    logger.info(f"Running spatial GEV analysis for {variable}...")

    for i in range(nlat):
        for j in range(nlon):
            ams = da.values[:, i, j]
            ams = ams[~np.isnan(ams) & (ams > 0)]

            if len(ams) < 10:
                continue

            try:
                params = fit_gev_mle(ams)
                loc, scale, shape = params["loc"], params["scale"], params["shape"]
                gev_loc[i, j] = loc
                gev_scale[i, j] = scale
                gev_shape[i, j] = shape

                rls = compute_return_levels(loc, scale, shape, return_periods)
                for T in return_periods:
                    rl_data[T][i, j] = rls[T]

                ci = bootstrap_return_levels(ams, return_periods, n_bootstrap, method="mle")
                for T in return_periods:
                    rl_lower[T][i, j] = ci[T][0]
                    rl_upper[T][i, j] = ci[T][2]

                for name, thresh in JAKARTA_FLOOD_THRESHOLDS.items():
                    exceedance[name][i, j] = exceedance_probability(loc, scale, shape, thresh)

            except Exception:
                pass

        if i % 3 == 0:
            logger.info(f"  Row {i+1}/{nlat}")

    coords = {"lat": ds_indices.lat, "lon": ds_indices.lon}
    dims = ["lat", "lon"]

    data_vars = {
        f"{variable}_GEV_loc": xr.DataArray(gev_loc, coords=coords, dims=dims,
                                             attrs={"units": "mm", "long_name": "GEV Location (μ)"}),
        f"{variable}_GEV_scale": xr.DataArray(gev_scale, coords=coords, dims=dims,
                                               attrs={"units": "mm", "long_name": "GEV Scale (σ)"}),
        f"{variable}_GEV_shape": xr.DataArray(gev_shape, coords=coords, dims=dims,
                                               attrs={"units": "-", "long_name": "GEV Shape (ξ)"}),
    }

    for T in return_periods:
        data_vars[f"{variable}_RL{T}yr"] = xr.DataArray(
            rl_data[T], coords=coords, dims=dims,
            attrs={"units": "mm", "long_name": f"{T}-year return level of {variable}"}
        )
        data_vars[f"{variable}_RL{T}yr_lower95"] = xr.DataArray(
            rl_lower[T], coords=coords, dims=dims,
            attrs={"units": "mm", "long_name": f"{T}-year RL lower 95% CI"}
        )
        data_vars[f"{variable}_RL{T}yr_upper95"] = xr.DataArray(
            rl_upper[T], coords=coords, dims=dims,
            attrs={"units": "mm", "long_name": f"{T}-year RL upper 95% CI"}
        )

    for name, thresh in JAKARTA_FLOOD_THRESHOLDS.items():
        data_vars[f"{variable}_exceedance_{name}"] = xr.DataArray(
            exceedance[name], coords=coords, dims=dims,
            attrs={"units": "probability/year",
                   "threshold_mm": thresh,
                   "long_name": f"Annual exceedance probability for {thresh} mm/day ({name})"}
        )

    return xr.Dataset(data_vars, attrs={
        "variable": variable,
        "distribution": "GEV (MLE)",
        "return_periods": str(return_periods),
        "bootstrap_samples": n_bootstrap,
    })


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--indices-dir", default="results/indices", show_default=True)
@click.option("--output-dir", default="results/extreme_freq", show_default=True)
@click.option("--scenario", default="all",
              type=click.Choice(["historical", "rcp26", "rcp45", "rcp85", "all"]))
@click.option("--variable", default="Rx1day",
              type=click.Choice(["Rx1day", "Rx3day", "Rx5day"]))
@click.option("--n-bootstrap", default=500, show_default=True)
def main(indices_dir, output_dir, scenario, variable, n_bootstrap):
    """Run GEV extreme rainfall frequency analysis."""
    indices_path = Path(indices_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenarios = ["historical", "rcp26", "rcp45", "rcp85"] if scenario == "all" else [scenario]

    for scen in scenarios:
        files = sorted(indices_path.glob(f"*HadGEM2-AO*{scen}*indices*.nc"))
        if not files:
            logger.warning(f"No indices for {scen}")
            continue

        logger.info(f"\nProcessing: {scen} — {variable}")
        ds = xr.open_dataset(files[0])
        ds_out = spatial_frequency_analysis(ds, variable=variable, n_bootstrap=n_bootstrap)
        ds_out.attrs["scenario"] = scen

        out_file = out_path / f"HadGEM2-AO_{scen}_{variable}_extreme_freq_jakarta.nc"
        ds_out.to_netcdf(out_file)
        logger.info(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
