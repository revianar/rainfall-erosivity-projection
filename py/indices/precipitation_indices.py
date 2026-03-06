import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WET_DAY_THRESHOLD = 1.0  # mm/day


# ── Core index functions ──────────────────────────────────────────────────────

def compute_prcptot(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_pr = pr.where(pr >= threshold, 0.0)
    prcptot = wet_pr.resample(time="YE").sum(dim="time")
    prcptot.attrs = {
        "long_name": "Annual Total Wet-Day Precipitation",
        "standard_name": "PRCPTOT",
        "units": "mm/year",
        "description": f"Sum of daily precip on days with pr >= {threshold} mm/day",
    }
    return prcptot


def compute_rxnday(pr: xr.DataArray, n: int = 1) -> xr.DataArray:
    if n == 1:
        rolling_sum = pr
    else:
        rolling_sum = pr.rolling(time=n, center=False).sum()

    rxn = rolling_sum.resample(time="YE").max(dim="time")
    rxn.attrs = {
        "long_name": f"Annual Maximum {n}-Day Precipitation",
        "standard_name": f"Rx{n}day",
        "units": "mm",
        "description": f"Maximum {n}-day accumulated precipitation per year",
    }
    return rxn


def compute_wdf(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_days = (pr >= threshold).astype(float)
    wdf = wet_days.resample(time="YE").sum(dim="time")
    wdf.attrs = {
        "long_name": "Annual Wet-Day Frequency",
        "standard_name": "WDF",
        "units": "days/year",
        "description": f"Count of days with pr >= {threshold} mm/day",
    }
    return wdf


def compute_sdii(pr: xr.DataArray, threshold: float = WET_DAY_THRESHOLD) -> xr.DataArray:
    wet_pr = pr.where(pr >= threshold, 0.0)
    annual_total = wet_pr.resample(time="YE").sum(dim="time")

    wet_days = (pr >= threshold).astype(float)
    annual_wdf = wet_days.resample(time="YE").sum(dim="time")

    sdii = annual_total / annual_wdf.where(annual_wdf > 0)
    sdii.attrs = {
        "long_name": "Simple Daily Intensity Index",
        "standard_name": "SDII",
        "units": "mm/wet-day",
        "description": "Annual total precip divided by number of wet days",
    }
    return sdii


def compute_all_indices(pr: xr.DataArray) -> xr.Dataset:
    logger.info("Computing PRCPTOT...")
    prcptot = compute_prcptot(pr)

    logger.info("Computing Rx1day...")
    rx1day = compute_rxnday(pr, n=1)

    logger.info("Computing Rx3day...")
    rx3day = compute_rxnday(pr, n=3)

    logger.info("Computing Rx5day...")
    rx5day = compute_rxnday(pr, n=5)

    logger.info("Computing WDF...")
    wdf = compute_wdf(pr)

    logger.info("Computing SDII...")
    sdii = compute_sdii(pr)

    # Rename time coordinate to 'year' for clarity
    def to_year(da):
        da["time"] = da["time"].dt.year
        return da.rename({"time": "year"})

    ds = xr.Dataset({
        "PRCPTOT": to_year(prcptot),
        "Rx1day":  to_year(rx1day),
        "Rx3day":  to_year(rx3day),
        "Rx5day":  to_year(rx5day),
        "WDF":     to_year(wdf),
        "SDII":    to_year(sdii),
    })

    ds.attrs = {
        "title": "ETCCDI Precipitation Indices — Jakarta Greater Capital Region",
        "model": "HadGEM2-AO",
        "wet_day_threshold": f"{WET_DAY_THRESHOLD} mm/day",
    }
    return ds


def compute_period_mean(ds_indices: xr.Dataset, period: tuple) -> xr.Dataset:
    start, end = period
    ds_period = ds_indices.sel(year=slice(start, end))
    ds_mean = ds_period.mean(dim="year")
    ds_mean.attrs["period"] = f"{start}–{end}"
    return ds_mean


def compute_change_signal(
    ds_future: xr.Dataset,
    ds_historical: xr.Dataset,
    method: str = "absolute",
) -> xr.Dataset:
    if method == "absolute":
        delta = ds_future - ds_historical
        delta.attrs["change_method"] = "absolute (future − historical)"
    elif method == "relative":
        delta = ((ds_future - ds_historical) / ds_historical) * 100.0
        delta.attrs["change_method"] = "relative (% change)"
        for var in delta.data_vars:
            delta[var].attrs["units"] = "%"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'absolute' or 'relative'.")
    return delta


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--input-dir", default="data/bias_corrected", show_default=True)
@click.option("--output-dir", default="results/indices", show_default=True)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(["historical", "rcp26", "rcp45", "rcp85", "all"]),
    show_default=True,
)
def main(input_dir: str, output_dir: str, scenario: str):
    """Compute annual precipitation indices from bias-corrected CMIP5 data."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenarios = ["historical", "rcp26", "rcp45", "rcp85"] if scenario == "all" else [scenario]

    for scen in scenarios:
        pattern = f"*HadGEM2-AO*{scen}*jakarta*.nc"
        files = sorted(input_path.glob(pattern))
        if not files:
            logger.warning(f"No bias-corrected files for scenario: {scen}")
            continue

        logger.info(f"\n{'='*50}\nProcessing scenario: {scen}\n{'='*50}")
        ds = xr.open_mfdataset(files, combine="by_coords", use_cftime=True)
        pr = ds["pr"]

        indices = compute_all_indices(pr)
        indices.attrs["scenario"] = scen

        out_file = output_path / f"HadGEM2-AO_{scen}_indices_jakarta.nc"
        indices.to_netcdf(out_file)
        logger.info(f"Indices saved: {out_file}")

    logger.info("\nAll indices computed successfully.")


if __name__ == "__main__":
    main()
