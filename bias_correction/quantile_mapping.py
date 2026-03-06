import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d
from pathlib import Path
import pickle
import click
import logging
from typing import Tuple, Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WET_DAY_THRESHOLD = 1.0  # mm/day
N_QUANTILES = 100


# Empirical QM ====================

def fit_eqm_transfer(
    obs: np.ndarray,
    hist: np.ndarray,
    n_quantiles: int = N_QUANTILES,
    threshold: float = WET_DAY_THRESHOLD,
) -> dict:
    quantiles = np.linspace(0, 1, n_quantiles + 1)

    obs_wet = obs[obs >= threshold]
    hist_wet = hist[hist >= threshold]

    obs_cdf = np.quantile(obs_wet, quantiles) if len(obs_wet) > 0 else np.zeros(len(quantiles))
    hist_cdf = np.quantile(hist_wet, quantiles) if len(hist_wet) > 0 else np.zeros(len(quantiles))

    return {
        "method": "empirical",
        "quantiles": quantiles,
        "obs_cdf": obs_cdf,
        "hist_cdf": hist_cdf,
        "wet_prob_obs": len(obs_wet) / len(obs[~np.isnan(obs)]),
        "wet_prob_hist": len(hist_wet) / len(hist[~np.isnan(hist)]),
        "threshold": threshold,
    }


def apply_eqm(
    data: np.ndarray,
    transfer: dict,
    preserve_dry_days: bool = True,
) -> np.ndarray:
    corrected = data.copy()
    threshold = transfer["threshold"]
    wet_mask = data >= threshold

    if preserve_dry_days:
        # Adjust wet-day frequency by comparing wet-day probability
        prob_ratio = transfer["wet_prob_obs"] / max(transfer["wet_prob_hist"], 1e-6)
        if prob_ratio < 1.0:
            # Model has too many wet days — convert some to dry
            wet_indices = np.where(wet_mask)[0]
            n_to_dry = int(len(wet_indices) * (1 - prob_ratio))
            # Remove driest wet days first
            if n_to_dry > 0:
                wet_precip = data[wet_indices]
                to_dry = wet_indices[np.argsort(wet_precip)[:n_to_dry]]
                corrected[to_dry] = 0.0
                wet_mask[to_dry] = False

    # Quantile mapping for wet days
    if wet_mask.sum() > 0:
        wet_data = data[wet_mask]
        # Find quantile of each model wet-day value in hist distribution
        hist_cdf = transfer["hist_cdf"]
        obs_cdf = transfer["obs_cdf"]

        # Interpolate: model value → quantile → obs value
        f_interp = interp1d(
            hist_cdf, obs_cdf,
            bounds_error=False,
            fill_value=(obs_cdf[0], obs_cdf[-1])
        )
        corrected[wet_mask] = f_interp(wet_data)
        corrected[wet_mask] = np.maximum(corrected[wet_mask], threshold)

    corrected[~wet_mask & (corrected > 0)] = 0.0
    corrected = np.maximum(corrected, 0.0)
    return corrected


# Parametric QM (Gamma) ====================

def fit_pqm_transfer(
    obs: np.ndarray,
    hist: np.ndarray,
    threshold: float = WET_DAY_THRESHOLD,
) -> dict:
    obs_wet = obs[obs >= threshold]
    hist_wet = hist[hist >= threshold]

    obs_params = stats.gamma.fit(obs_wet, floc=0) if len(obs_wet) > 10 else (1, 0, 1)
    hist_params = stats.gamma.fit(hist_wet, floc=0) if len(hist_wet) > 10 else (1, 0, 1)

    return {
        "method": "parametric_gamma",
        "obs_gamma": obs_params,   # (shape, loc, scale)
        "hist_gamma": hist_params,
        "wet_prob_obs": len(obs_wet) / len(obs[~np.isnan(obs)]),
        "wet_prob_hist": len(hist_wet) / len(hist[~np.isnan(hist)]),
        "threshold": threshold,
    }


def apply_pqm(data: np.ndarray, transfer: dict) -> np.ndarray:
    corrected = data.copy()
    threshold = transfer["threshold"]
    wet_mask = data >= threshold

    if wet_mask.sum() > 0:
        wet_data = data[wet_mask]
        obs_shape, obs_loc, obs_scale = transfer["obs_gamma"]
        hist_shape, hist_loc, hist_scale = transfer["hist_gamma"]

        # Quantile in historical distribution
        p = stats.gamma.cdf(wet_data, hist_shape, loc=hist_loc, scale=hist_scale)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Map to observational distribution
        corrected[wet_mask] = stats.gamma.ppf(p, obs_shape, loc=obs_loc, scale=obs_scale)
        corrected[wet_mask] = np.maximum(corrected[wet_mask], threshold)

    corrected[~wet_mask] = 0.0
    return corrected


# Spatial Application ====================

def apply_bias_correction_spatial(
    obs_ds: xr.Dataset,
    hist_ds: xr.Dataset,
    future_ds: xr.Dataset,
    method: Literal["empirical", "parametric"] = "empirical",
    pr_var: str = "pr",
) -> xr.DataArray:
    obs_pr = obs_ds[pr_var].values
    hist_pr = hist_ds[pr_var].values
    future_pr = future_ds[pr_var].values

    corrected = np.full_like(future_pr, np.nan)
    ntime, nlat, nlon = future_pr.shape
    transfer_functions = {}

    logger.info(f"Applying {method} QM over {nlat}×{nlon} grid cells...")

    for i in range(nlat):
        for j in range(nlon):
            obs_1d = obs_pr[:, i, j]
            hist_1d = hist_pr[:, i, j]
            future_1d = future_pr[:, i, j]

            if np.all(np.isnan(obs_1d)) or np.all(np.isnan(hist_1d)):
                continue

            if method == "empirical":
                transfer = fit_eqm_transfer(obs_1d, hist_1d)
                corrected[:, i, j] = apply_eqm(future_1d, transfer)
            else:
                transfer = fit_pqm_transfer(obs_1d, hist_1d)
                corrected[:, i, j] = apply_pqm(future_1d, transfer)

            transfer_functions[(i, j)] = transfer

        if i % 5 == 0:
            logger.info(f"  Row {i+1}/{nlat} complete")

    da_corrected = xr.DataArray(
        corrected,
        coords=future_ds[pr_var].coords,
        dims=future_ds[pr_var].dims,
        attrs={
            **future_ds[pr_var].attrs,
            "bias_correction": f"Quantile Mapping ({method})",
            "reference_obs": "CHIRPS v2.0",
            "units": "mm/day",
        },
    )
    return da_corrected, transfer_functions


# CLI ====================

@click.command()
@click.option("--obs", required=True, help="Path to observational NetCDF (CHIRPS)")
@click.option("--hist", required=True, help="Path to historical model NetCDF")
@click.option("--future", required=True, help="Path to future model NetCDF to correct")
@click.option("--output-dir", default="data/bias_corrected", show_default=True)
@click.option("--scenario", default="rcp85", show_default=True)
@click.option("--method", default="empirical",
              type=click.Choice(["empirical", "parametric"]), show_default=True)
@click.option("--save-transfer", is_flag=True, default=False,
              help="Save transfer functions as pickle")
def main(obs, hist, future, output_dir, scenario, method, save_transfer):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading datasets...")
    obs_ds = xr.open_dataset(obs, use_cftime=True)
    hist_ds = xr.open_dataset(hist, use_cftime=True)
    future_ds = xr.open_dataset(future, use_cftime=True)

    # Align temporal overlap between obs and hist
    obs_years = obs_ds.time.dt.year.values
    hist_years = hist_ds.time.dt.year.values
    common_years = np.intersect1d(obs_years, hist_years)
    logger.info(f"Common calibration period: {common_years[0]}–{common_years[-1]}")

    obs_cal = obs_ds.sel(time=obs_ds.time.dt.year.isin(common_years))
    hist_cal = hist_ds.sel(time=hist_ds.time.dt.year.isin(common_years))

    corrected, transfers = apply_bias_correction_spatial(
        obs_cal, hist_cal, future_ds, method=method
    )

    out_file = out_path / f"HadGEM2-AO_{scenario}_pr_bc_{method}_jakarta.nc"
    corrected.to_netcdf(out_file)
    logger.info(f"Bias-corrected data saved: {out_file}")

    if save_transfer:
        tf_file = out_path / f"transfer_functions_{scenario}_{method}.pkl"
        with open(tf_file, "wb") as f:
            pickle.dump(transfers, f)
        logger.info(f"Transfer functions saved: {tf_file}")


if __name__ == "__main__":
    main()
