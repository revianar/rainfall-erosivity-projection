import sys
import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats
from pathlib import Path
import logging
import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDICES_DIR = PROJECT_ROOT / "py" / "results" / "indices"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "py" / "results" / "extreme_freq"

MODEL    = "HadGEM2-AO"
SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]

# Return periods to compute [years]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100]

# Jakarta flood-relevant thresholds [mm/day]
JAKARTA_FLOOD_THRESHOLDS = {
    "moderate_flood": 100.0,   # widespread moderate flooding
    "severe_flood":   150.0,   # severe inundation events
    "extreme_flood":  200.0,   # catastrophic events (e.g. 2007, 2020)
}


# ===== L-moments ==============================================================

def lmoments(x: np.ndarray) -> tuple:
    n        = len(x)
    x_sorted = np.sort(x)

    # Probability-weighted moments using vectorised computation
    b = np.zeros(4)
    idx = np.arange(n)
    for r in range(4):
        if r == 0:
            weights = np.ones(n) / n
        else:
            # w_j = C(j, r) / C(n-1, r)  for j = 0 … n-1
            num = np.array([
                np.prod([idx[j] - k for k in range(r)]) for j in range(n)
            ], dtype=float)
            den = np.prod([n - 1 - k for k in range(r)], dtype=float)
            weights = num / den / n
        b[r] = np.dot(x_sorted, weights)

    l1 = b[0]
    l2 = 2 * b[1] - b[0]
    l3 = 6 * b[2] - 6 * b[1] + b[0]
    l4 = 20 * b[3] - 30 * b[2] + 12 * b[1] - b[0]
    return l1, l2, l3, l4


# ===== GEV fitting ============================================================

def fit_gev_lmoments(ams: np.ndarray) -> dict:
    l1, l2, l3, _ = lmoments(ams)
    t3 = l3 / l2   # L-skewness

    if abs(t3) < 1e-6:
        shape = 0.0
    else:
        c     = 2.0 / (3.0 + t3) - np.log(2) / np.log(3)
        shape = 7.8590 * c + 2.9554 * c ** 2

    if abs(shape) < 1e-6:   # Gumbel limit
        scale = l2 / np.log(2)
        loc   = l1 - 0.5772 * scale
    else:
        g     = stats.gamma(1 + shape)          # Gamma(1 + ξ) as a scalar
        scale = shape * l2 / ((1 - 2 ** (-shape)) * g)
        loc   = l1 - scale * (1 - g) / shape

    return {"loc": float(loc), "scale": float(scale), "shape": float(shape),
            "method": "L-moments"}


def fit_gev_mle(ams: np.ndarray) -> dict:
    shape_scipy, loc, scale = stats.genextreme.fit(ams)
    nll = -np.sum(stats.genextreme.logpdf(ams, shape_scipy, loc=loc, scale=scale))
    return {"loc": float(loc), "scale": float(scale), "shape": float(-shape_scipy),
            "method": "MLE", "nll": float(nll)}


def compute_return_levels(
    loc: float,
    scale: float,
    shape: float,
    return_periods: list = RETURN_PERIODS,
) -> dict:
    """Compute GEV return levels for given return periods."""
    return_levels = {}
    for T in return_periods:
        p  = 1 - 1.0 / T      # non-exceedance probability
        yT = -np.log(p)        # reduced variate (Gumbel)

        if abs(shape) < 1e-6:  # Gumbel limit
            xT = loc - scale * np.log(yT)
        else:
            xT = loc - (scale / shape) * (1 - yT ** (-shape))

        return_levels[T] = float(xT)
    return return_levels


def bootstrap_return_levels(
    ams: np.ndarray,
    return_periods: list = RETURN_PERIODS,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    method: str = "mle",
) -> dict:
    n          = len(ams)
    rl_samples = {T: [] for T in return_periods}

    for _ in range(n_bootstrap):
        sample = np.random.choice(ams, size=n, replace=True)
        try:
            params = fit_gev_mle(sample) if method == "mle" else fit_gev_lmoments(sample)
            rls    = compute_return_levels(
                params["loc"], params["scale"], params["shape"], return_periods
            )
            for T, rl in rls.items():
                rl_samples[T].append(rl)
        except Exception:
            pass

    alpha      = (1 - ci) / 2
    ci_results = {}
    for T in return_periods:
        arr = np.array(rl_samples[T])
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            ci_results[T] = (
                float(np.percentile(arr, alpha * 100)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, (1 - alpha) * 100)),
            )
        else:
            ci_results[T] = (np.nan, np.nan, np.nan)
    return ci_results


# ===== Threshold exceedance ===================================================

def exceedance_probability(
    loc: float,
    scale: float,
    shape: float,
    threshold: float,
) -> float:
    """Annual exceedance probability P(X > threshold) from fitted GEV."""
    if abs(shape) < 1e-6:
        p_non_exceed = np.exp(-np.exp(-(threshold - loc) / scale))
    else:
        z = 1 + shape * (threshold - loc) / scale
        if z <= 0:
            return 1.0
        p_non_exceed = np.exp(-(z ** (-1.0 / shape)))
    return float(1 - p_non_exceed)


# ===== Core GEV analysis (works for 1-D and spatial data) ====================

def _analyse_ams(
    ams: np.ndarray,
    return_periods: list,
    n_bootstrap: int,
) -> dict | None:
    ams = ams[np.isfinite(ams) & (ams > 0)]
    if len(ams) < 10:
        return None

    try:
        params = fit_gev_mle(ams)
        loc, scale, shape = params["loc"], params["scale"], params["shape"]

        rls = compute_return_levels(loc, scale, shape, return_periods)
        ci  = bootstrap_return_levels(ams, return_periods, n_bootstrap, method="mle")

        exceedance = {
            name: exceedance_probability(loc, scale, shape, thresh)
            for name, thresh in JAKARTA_FLOOD_THRESHOLDS.items()
        }

        return {
            "loc": loc, "scale": scale, "shape": shape,
            "return_levels": rls,
            "ci": ci,
            "exceedance": exceedance,
        }
    except Exception:
        return None


def spatial_frequency_analysis(
    ds_indices: xr.Dataset,
    variable: str = "Rx1day",
    return_periods: list = RETURN_PERIODS,
    n_bootstrap: int = 500,
) -> xr.Dataset:
    da = ds_indices[variable]

    # Determine spatial structure
    has_lat = "lat" in da.dims
    has_lon = "lon" in da.dims

    if has_lat and has_lon:
        nlat = len(da.lat)
        nlon = len(da.lon)
        coords = {"lat": da.lat, "lon": da.lon}
        dims   = ["lat", "lon"]
        shape_2d = (nlat, nlon)
    else:
        # Single grid cell — treat as (1, 1) for uniform code path, unwrap at end
        nlat, nlon = 1, 1
        coords = {}
        dims   = []
        shape_2d = (1, 1)

    # Output arrays
    gev_loc   = np.full(shape_2d, np.nan)
    gev_scale = np.full(shape_2d, np.nan)
    gev_shape = np.full(shape_2d, np.nan)
    rl_data   = {T: np.full(shape_2d, np.nan) for T in return_periods}
    rl_lower  = {T: np.full(shape_2d, np.nan) for T in return_periods}
    rl_upper  = {T: np.full(shape_2d, np.nan) for T in return_periods}
    exceed    = {k: np.full(shape_2d, np.nan) for k in JAKARTA_FLOOD_THRESHOLDS}

    logger.info(f"  Running GEV analysis for {variable} "
                f"({'spatial' if has_lat else 'single cell'})...")

    for i in range(nlat):
        for j in range(nlon):
            if has_lat and has_lon:
                ams = da.values[:, i, j]
            else:
                ams = da.values   # (year,)

            result = _analyse_ams(ams, return_periods, n_bootstrap)
            if result is None:
                continue

            gev_loc[i, j]   = result["loc"]
            gev_scale[i, j] = result["scale"]
            gev_shape[i, j] = result["shape"]

            for T in return_periods:
                rl_data[T][i, j]  = result["return_levels"][T]
                rl_lower[T][i, j] = result["ci"][T][0]
                rl_upper[T][i, j] = result["ci"][T][2]

            for name in JAKARTA_FLOOD_THRESHOLDS:
                exceed[name][i, j] = result["exceedance"][name]

        if has_lat and i % 3 == 0:
            logger.info(f"    Row {i+1}/{nlat}")

    # Build output Dataset
    def _da(arr, long_name, units):
        if has_lat and has_lon:
            return xr.DataArray(arr, coords=coords, dims=dims,
                                attrs={"long_name": long_name, "units": units})
        else:
            # Return scalar DataArray for single-cell case
            return xr.DataArray(float(arr[0, 0]),
                                attrs={"long_name": long_name, "units": units})

    data_vars = {
        f"{variable}_GEV_loc":   _da(gev_loc,   "GEV Location (mu)",  "mm"),
        f"{variable}_GEV_scale": _da(gev_scale, "GEV Scale (sigma)",  "mm"),
        f"{variable}_GEV_shape": _da(gev_shape, "GEV Shape (xi)",     "-"),
    }

    for T in return_periods:
        data_vars[f"{variable}_RL{T}yr"] = _da(
            rl_data[T], f"{T}-year return level of {variable}", "mm"
        )
        data_vars[f"{variable}_RL{T}yr_lower95"] = _da(
            rl_lower[T], f"{T}-year RL lower 95% CI", "mm"
        )
        data_vars[f"{variable}_RL{T}yr_upper95"] = _da(
            rl_upper[T], f"{T}-year RL upper 95% CI", "mm"
        )

    for name, thresh in JAKARTA_FLOOD_THRESHOLDS.items():
        data_vars[f"{variable}_exceedance_{name}"] = _da(
            exceed[name],
            f"Annual exceedance probability for {thresh} mm/day ({name})",
            "probability/year",
        )

    ds_out = xr.Dataset(data_vars, attrs={
        "variable":          variable,
        "distribution":      "GEV (MLE)",
        "return_periods":    str(return_periods),
        "bootstrap_samples": n_bootstrap,
    })
    return ds_out


# ===== CLI ====================================================================

@click.command()
@click.option(
    "--indices-dir",
    default=str(DEFAULT_INDICES_DIR),
    show_default=True,
    help="Directory containing index NetCDF files (results/indices/).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save extreme frequency output files.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(SCENARIOS + ["all"]),
    show_default=True,
    help="Scenario to process, or 'all' for every scenario.",
)
@click.option(
    "--variable",
    default="Rx1day",
    type=click.Choice(["Rx1day", "Rx3day", "Rx5day"]),
    show_default=True,
    help="Annual maxima variable to analyse.",
)
@click.option(
    "--n-bootstrap",
    default=500,
    show_default=True,
    help="Number of bootstrap resamples for confidence intervals.",
)
def main(indices_dir, output_dir, scenario, variable, n_bootstrap):
    indices_path = Path(indices_dir)
    out_path     = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not indices_path.exists():
        logger.error(f"Indices directory not found: {indices_path}")
        logger.error("Run precipitation_indices.py --scenario all first.")
        sys.exit(1)

    scenarios_to_run = SCENARIOS if scenario == "all" else [scenario]

    completed = []
    failed    = []

    for scen in scenarios_to_run:
        # Exact filename produced by precipitation_indices.py
        fname = f"{MODEL}_{scen}_indices_jakarta.nc"
        fpath = indices_path / fname

        if not fpath.exists():
            logger.warning(f"  File not found: {fpath} — skipping {scen}")
            failed.append(scen)
            continue

        logger.info(f"{'='*55}")
        logger.info(f"Processing: {scen} — {variable}")
        logger.info(f"{'='*55}")

        ds = xr.open_dataset(fpath)

        if variable not in ds:
            logger.error(f"  Variable '{variable}' not found in {fname}")
            logger.error(f"  Available: {list(ds.data_vars)}")
            failed.append(scen)
            ds.close()
            continue

        n_years = len(ds["year"]) if "year" in ds.coords else len(ds[variable])
        logger.info(f"  Years in record : {n_years}")
        logger.info(f"  Bootstrap samples: {n_bootstrap}")

        ds_out = spatial_frequency_analysis(
            ds,
            variable     = variable,
            return_periods = RETURN_PERIODS,
            n_bootstrap  = n_bootstrap,
        )
        ds_out.attrs["scenario"] = scen

        out_file = out_path / f"{MODEL}_{scen}_{variable}_extreme_freq_jakarta.nc"
        ds_out.to_netcdf(
            out_file,
            encoding={v: {"dtype": "float32"} for v in ds_out.data_vars
                      if ds_out[v].ndim > 0},
        )
        logger.info(f"  Saved: {out_file.name}")

        # Quick summary of key return levels
        for T in [10, 50, 100]:
            key = f"{variable}_RL{T}yr"
            if key in ds_out:
                val = ds_out[key].values
                mean_rl = float(np.nanmean(val))
                logger.info(f"  {T}-yr return level (mean): {mean_rl:.1f} mm")

        ds.close()
        completed.append(scen)

    logger.info(f"{'='*55}")
    logger.info(f"DONE  —  {len(completed)} scenario(s) processed")
    logger.info(f"{'='*55}")
    for scen in completed:
        out = out_path / f"{MODEL}_{scen}_{variable}_extreme_freq_jakarta.nc"
        logger.info(f"  {scen:<12} -> {out.name}")
    if failed:
        logger.warning(f"  Skipped : {failed}")
    logger.info(f"\nOutput directory: {out_path}")


if __name__ == "__main__":
    main()