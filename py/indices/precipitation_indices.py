import sys
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_DIR  = PROJECT_ROOT / "py" / "esgf" / "bias_corrected"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "py" / "results" / "indices"

MODEL    = "HadGEM2-AO"
ENSEMBLE = "r1i1p1"
SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]

WET_DAY_THRESHOLD = 1.0           # mm/day
QM_METHOD         = "empirical"   # must match the suffix used by quantile_mapping.py


# ===== Safe year extraction ====================

def _get_years(da: xr.DataArray) -> np.ndarray:
    """
    Extract integer years from a time coordinate.
    Works with both numpy datetime64 and cftime objects (e.g. Datetime360Day).
    """
    time_vals = da.time.values
    if hasattr(time_vals[0], "year"):
        # cftime objects — each has a .year attribute
        return np.array([t.year for t in time_vals])
    else:
        # numpy datetime64 — use pandas
        return pd.DatetimeIndex(time_vals).year.values


def _resample_yearly(da: xr.DataArray, func: str) -> xr.DataArray:
    """
    Resample a DataArray to yearly frequency using func ('sum' or 'max').
    Uses groupby on cftime-safe integer years instead of .resample(time='YE')
    which can fail on 360-day calendars.
    """
    years = _get_years(da)
    unique_years = np.unique(years)

    slices = []
    for yr in unique_years:
        da_yr = da.isel(time=np.where(years == yr)[0])
        if func == "sum":
            slices.append(da_yr.sum(dim="time"))
        elif func == "max":
            slices.append(da_yr.max(dim="time"))
        else:
            raise ValueError(f"Unknown func: {func}")

    result = xr.concat(slices, dim=pd.Index(unique_years, name="year"))
    return result


# ===== Index computation functions ============================================

def compute_prcptot(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_pr  = pr.where(pr >= threshold, 0.0)
    prcptot = _resample_yearly(wet_pr, "sum")
    prcptot.attrs = {
        "long_name":   "Annual Total Wet-Day Precipitation",
        "standard_name": "PRCPTOT",
        "units":       "mm/year",
        "description": f"Sum of daily precip on days with pr >= {threshold} mm/day",
    }
    return prcptot


def compute_rxnday(pr: xr.DataArray, n: int = 1) -> xr.DataArray:
    if n == 1:
        rolling_sum = pr
    else:
        rolling_sum = pr.rolling(time=n, center=False, min_periods=n).sum()

    rxn = _resample_yearly(rolling_sum, "max")
    rxn.attrs = {
        "long_name":   f"Annual Maximum {n}-Day Precipitation",
        "standard_name": f"Rx{n}day",
        "units":       "mm",
        "description": f"Maximum {n}-day accumulated precipitation per year",
    }
    return rxn


def compute_wdf(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_days = (pr >= threshold).astype(float)
    wdf      = _resample_yearly(wet_days, "sum")
    wdf.attrs = {
        "long_name":   "Annual Wet-Day Frequency",
        "standard_name": "WDF",
        "units":       "days/year",
        "description": f"Count of days with pr >= {threshold} mm/day",
    }
    return wdf


def compute_sdii(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_pr       = pr.where(pr >= threshold, 0.0)
    annual_total = _resample_yearly(wet_pr,  "sum")

    wet_days     = (pr >= threshold).astype(float)
    annual_wdf   = _resample_yearly(wet_days, "sum")

    sdii = annual_total / annual_wdf.where(annual_wdf > 0)
    sdii.attrs = {
        "long_name":   "Simple Daily Intensity Index",
        "standard_name": "SDII",
        "units":       "mm/wet-day",
        "description": "Annual total precip divided by number of wet days",
    }
    return sdii


def compute_all_indices(pr: xr.DataArray) -> xr.Dataset:
    logger.info("  Computing PRCPTOT...")
    prcptot = compute_prcptot(pr)

    logger.info("  Computing Rx1day...")
    rx1day  = compute_rxnday(pr, n=1)

    logger.info("  Computing Rx3day...")
    rx3day  = compute_rxnday(pr, n=3)

    logger.info("  Computing Rx5day...")
    rx5day  = compute_rxnday(pr, n=5)

    logger.info("  Computing WDF...")
    wdf     = compute_wdf(pr)

    logger.info("  Computing SDII...")
    sdii    = compute_sdii(pr)

    ds = xr.Dataset({
        "PRCPTOT": prcptot,
        "Rx1day":  rx1day,
        "Rx3day":  rx3day,
        "Rx5day":  rx5day,
        "WDF":     wdf,
        "SDII":    sdii,
    })

    ds.attrs = {
        "title":             "ETCCDI Precipitation Indices — Jakarta Greater Capital Region",
        "model":             MODEL,
        "wet_day_threshold": f"{WET_DAY_THRESHOLD} mm/day",
    }
    return ds


# ===== Period mean and change signal ==========================================

def compute_period_mean(ds_indices: xr.Dataset, period: tuple) -> xr.Dataset:
    """Mean of all indices over a given year range."""
    start, end   = period
    ds_period    = ds_indices.sel(year=slice(start, end))
    ds_mean      = ds_period.mean(dim="year")
    ds_mean.attrs["period"] = f"{start}-{end}"
    return ds_mean


def compute_change_signal(
    ds_future: xr.Dataset,
    ds_historical: xr.Dataset,
    method: str = "absolute",
) -> xr.Dataset:
    if method == "absolute":
        delta = ds_future - ds_historical
        delta.attrs["change_method"] = "absolute (future - historical)"
    elif method == "relative":
        delta = ((ds_future - ds_historical) / ds_historical) * 100.0
        delta.attrs["change_method"] = "relative (% change)"
        for var in delta.data_vars:
            delta[var].attrs["units"] = "%"
    else:
        raise ValueError(f"Unknown method: '{method}'. Use 'absolute' or 'relative'.")
    return delta


# ===== CLI ====================================================================

@click.command()
@click.option(
    "--input-dir",
    default=str(DEFAULT_INPUT_DIR),
    show_default=True,
    help="Directory containing bias-corrected NetCDF files (py/esgf/bias_corrected/).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save computed index NetCDF files.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(SCENARIOS + ["all"]),
    show_default=True,
    help="Scenario to process, or 'all' for every scenario.",
)
@click.option(
    "--method",
    default=QM_METHOD,
    type=click.Choice(["empirical", "parametric"]),
    show_default=True,
    help="QM method suffix used in bias-corrected filenames.",
)
def main(input_dir: str, output_dir: str, scenario: str, method: str):
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.error("Run quantile_mapping.py apply --scenario all first.")
        sys.exit(1)

    scenarios_to_run = SCENARIOS if scenario == "all" else [scenario]

    completed = []
    failed    = []

    for scen in scenarios_to_run:
        logger.info(f"{'='*43}")
        logger.info(f"Processing: {scen}")
        logger.info(f"{'='*43}")

        # Resolve input filename
        if scen == "historical":
            fname = f"pr_day_{MODEL}_historical_{ENSEMBLE}_jakarta_bc_input.nc"
        else:
            fname = f"pr_day_{MODEL}_{scen}_{ENSEMBLE}_jakarta_bc_{method}.nc"

        fpath = input_path / fname

        if not fpath.exists():
            logger.warning(f"  File not found: {fpath}")
            logger.warning(f"  Skipping {scen}.")
            failed.append(scen)
            continue

        logger.info(f"  Loading: {fname}")
        ds = xr.open_dataset(fpath, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))

        # Determine precipitation variable name
        if "pr" in ds:
            pr = ds["pr"]
        elif "precip" in ds:
            pr = ds["precip"]
        else:
            logger.error(f"  No 'pr' or 'precip' variable found in {fname}")
            failed.append(scen)
            ds.close()
            continue

        # Convert units if still in kg m-2 s-1
        if pr.attrs.get("units", "") in ["kg m-2 s-1", "kg/m2/s"]:
            logger.info("  Converting units: kg m-2 s-1 -> mm/day")
            pr = pr * 86400.0
            pr.attrs["units"] = "mm/day"

        logger.info(
            f"  Period : {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}"
            f"  |  {len(ds.time):,} days"
        )

        # Compute indices
        try:
            indices = compute_all_indices(pr)
            indices.attrs["scenario"] = scen
            indices.attrs["source_file"] = fname
        except Exception as e:
            logger.error(f"  Failed to compute indices for {scen}: {e}")
            failed.append(scen)
            ds.close()
            continue

        # Save output
        out_file = output_path / f"{MODEL}_{scen}_indices_jakarta.nc"
        indices.to_netcdf(
            out_file,
            encoding={v: {"dtype": "float32"} for v in indices.data_vars},
        )
        logger.info(f"  Saved: {out_file.name}")

        # Quick summary
        years = indices.year.values
        logger.info(f"  Years  : {int(years[0])} -> {int(years[-1])}  ({len(years)} years)")
        for var in ["PRCPTOT", "Rx1day", "WDF", "SDII"]:
            vals = indices[var].values.astype(float)
            finite = vals[np.isfinite(vals)]
            if len(finite) > 0:
                logger.info(
                    f"  {var:<10}: mean={float(finite.mean()):.1f}  "
                    f"max={float(finite.max()):.1f}  "
                    f"[{indices[var].attrs.get('units', '')}]"
                )

        ds.close()
        completed.append(scen)

    # ===== Final summary ====================
    logger.info(f"{'='*43}")
    logger.info(f"DONE  —  {len(completed)} scenario(s) processed")
    logger.info(f"{'='*43}")
    for scen in completed:
        out = output_path / f"{MODEL}_{scen}_indices_jakarta.nc"
        logger.info(f"  {scen:<12} -> {out.name}")
    if failed:
        logger.warning(f"  Skipped : {failed}")
    logger.info(f"\nOutput directory: {output_path}")


if __name__ == "__main__":
    main()