import sys
import xarray as xr
import numpy as np
from xarray.coding.times import CFDatetimeCoder
from scipy import stats
from pathlib import Path
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BC_DIR        = PROJECT_ROOT / "py" / "esgf" / "bias_corrected"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "py" / "esgf" / "processed"
DEFAULT_OUTPUT_DIR    = PROJECT_ROOT / "py" / "results" / "water_stress"

MODEL    = "HadGEM2-AO"
ENSEMBLE = "r1i1p1"
SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]
QM_METHOD = "empirical"


# ===== Helpers ====================================================

def _get_years(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "year"):
        return np.array([v.year for v in t])
    import pandas as pd
    return pd.DatetimeIndex(t).year.values


def _get_months(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "month"):
        return np.array([v.month for v in t])
    import pandas as pd
    return pd.DatetimeIndex(t).month.values


def _get_doy(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "year"):
        # cftime objects
        if hasattr(t[0], "calendar") or type(t[0]).__name__ == "Datetime360Day":
            return np.array([(v.month - 1) * 30 + v.day for v in t])
        else:
            return np.array([v.timetuple().tm_yday for v in t])
    import pandas as pd
    return pd.DatetimeIndex(t).dayofyear.values


def _resample_yearly(da: xr.DataArray, func: str) -> xr.DataArray:
    years        = _get_years(da)
    unique_years = np.unique(years)
    slices = []
    for yr in unique_years:
        da_yr = da.isel(time=np.where(years == yr)[0])
        slices.append(da_yr.sum("time") if func == "sum" else da_yr.max("time"))
    import pandas as pd
    return xr.concat(slices, dim=pd.Index(unique_years, name="year"))


def _resample_monthly(da: xr.DataArray, func: str) -> xr.DataArray:
    years  = _get_years(da)
    months = _get_months(da)
    import pandas as pd

    ym_pairs    = sorted(set(zip(years.tolist(), months.tolist())))
    slices      = []
    time_labels = []

    for yr, mo in ym_pairs:
        mask  = (years == yr) & (months == mo)
        da_mo = da.isel(time=np.where(mask)[0])
        slices.append(da_mo.sum("time") if func == "sum" else da_mo.mean("time"))
        time_labels.append(pd.Timestamp(year=yr, month=mo, day=15))

    result = xr.concat(slices, dim=pd.Index(time_labels, name="time"))
    return result


# ===== Hargreaves-Samani PET ==================================================

def extraterrestrial_radiation(lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg)[:, np.newaxis]   # (nlat, 1)

    dr    = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)          # (ntime,)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)       # (ntime,)

    # Clamp argument to arccos to [-1, 1] to avoid NaN at extreme latitudes
    arg = np.clip(-np.tan(lat_rad) * np.tan(delta), -1.0, 1.0)
    ws  = np.arccos(arg)                                         # (nlat, ntime)

    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )
    return np.maximum(Ra, 0.0)   # (nlat, ntime)


def hargreaves_pet(
    tmax: xr.DataArray,
    tmin: xr.DataArray,
    lat: np.ndarray,
    doy: np.ndarray,
) -> xr.DataArray:
    """
    Hargreaves-Samani potential evapotranspiration [mm/day].

    PET = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)

    Parameters
    ----------
    tmax : xr.DataArray  (time, lat, lon)  °C
    tmin : xr.DataArray  (time, lat, lon)  °C
    lat  : np.ndarray    (nlat,)           degrees
    doy  : np.ndarray    (ntime,)          day-of-year
    """
    tmean = (tmax + tmin) / 2.0
    tdiff = (tmax - tmin).clip(min=0)

    Ra = extraterrestrial_radiation(lat, doy)       # (nlat, ntime)

    # Broadcast Ra to (time, lat, lon)
    nlon  = len(tmax.lon)
    Ra_T  = Ra.T                                    # (ntime, nlat)
    Ra_3d = Ra_T[:, :, np.newaxis] * np.ones(nlon)  # (ntime, nlat, nlon)

    Ra_da = xr.DataArray(
        Ra_3d,
        coords={"time": tmax.time, "lat": tmax.lat, "lon": tmax.lon},
        dims=["time", "lat", "lon"],
    )

    pet = 0.0023 * Ra_da * (tmean + 17.8) * (tdiff ** 0.5)
    pet = pet.clip(min=0)
    pet.attrs = {
        "units":     "mm/day",
        "long_name": "Potential Evapotranspiration (Hargreaves-Samani)",
    }
    return pet


# ===== Water stress metrics ===================================================

def compute_aridity_index(pr: xr.DataArray, pet: xr.DataArray) -> xr.DataArray:
    """
    UNEP Aridity Index: AI = MAP / MAE (annual totals) (Sahin, 2012).

    Classification:
      AI > 0.65        Humid
      0.50 – 0.65      Dry sub-humid
      0.20 – 0.50      Semi-arid
      0.05 – 0.20      Arid
      < 0.05           Hyper-arid
    """
    annual_pr  = _resample_yearly(pr,  "sum")
    annual_pet = _resample_yearly(pet, "sum")
    ai = annual_pr / annual_pet.where(annual_pet > 0)
    ai.attrs = {
        "long_name":   "Aridity Index (MAP/MAE)",
        "units":       "dimensionless",
        "description": "UNEP Aridity Index; AI > 0.65 = Humid, < 0.05 = Hyper-arid",
    }
    return ai


def compute_moisture_deficit(pr: xr.DataArray, pet: xr.DataArray) -> xr.DataArray:
    """
    Annual moisture deficit: PET - P [mm/year].
    Positive = water deficit.
    """
    annual_pr  = _resample_yearly(pr,  "sum")
    annual_pet = _resample_yearly(pet, "sum")
    deficit = annual_pet - annual_pr
    deficit.attrs = {
        "long_name":   "Annual Moisture Deficit (PET - P)",
        "units":       "mm/year",
        "description": "Positive = water deficit (demand > supply)",
    }
    return deficit


def compute_spi(pr_monthly: xr.DataArray, scale: int = 12) -> xr.DataArray:
    """
    Standardized Precipitation Index (SPI) at given accumulation scale.

    Interpretation:
      < -2.0  Extreme drought
      > +2.0  Extreme wet
    """
    logger.info(f"  Computing SPI-{scale}...")

    # Rolling accumulation along time
    pr_accum = pr_monthly.rolling(time=scale, center=False, min_periods=scale).sum()

    spi_values = np.full(pr_accum.shape, np.nan)

    if pr_accum.ndim == 1:
        # Single grid cell
        series = pr_accum.values
        valid  = series[~np.isnan(series) & (series > 0)]
        if len(valid) >= 20:
            try:
                shape, loc, sc = stats.gamma.fit(valid, floc=0)
                cdf   = stats.gamma.cdf(series, shape, loc=loc, scale=sc)
                p0    = np.sum(series == 0) / np.sum(~np.isnan(series))
                cdf   = p0 + (1 - p0) * cdf
                spi_values = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
            except Exception:
                pass
    else:
        nlat = len(pr_monthly.lat)
        nlon = len(pr_monthly.lon)
        for i in range(nlat):
            for j in range(nlon):
                series = pr_accum.values[:, i, j]
                valid  = series[~np.isnan(series) & (series > 0)]
                if len(valid) < 20:
                    continue
                try:
                    shape, loc, sc = stats.gamma.fit(valid, floc=0)
                    cdf   = stats.gamma.cdf(series, shape, loc=loc, scale=sc)
                    p0    = np.sum(series == 0) / np.sum(~np.isnan(series))
                    cdf   = p0 + (1 - p0) * cdf
                    spi_values[:, i, j] = stats.norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
                except Exception:
                    pass

    spi = xr.DataArray(
        spi_values,
        coords=pr_accum.coords,
        dims=pr_accum.dims,
        attrs={
            "long_name":   f"Standardized Precipitation Index (SPI-{scale})",
            "units":       "dimensionless",
            "scale":       scale,
            "description": "< -2.0: Extreme drought | > +2.0: Extreme wet",
        },
    )
    return spi


def compute_water_stress_months(
    pr_monthly: xr.DataArray,
    pet_monthly: xr.DataArray,
    threshold: float = 0.5,
) -> xr.DataArray:
    stressed     = (pr_monthly < threshold * pet_monthly).astype(float)
    years        = _get_years(stressed)
    unique_years = np.unique(years)
    import pandas as pd
    slices = [
        stressed.isel(time=np.where(years == yr)[0]).sum("time")
        for yr in unique_years
    ]
    annual_count = xr.concat(slices, dim=pd.Index(unique_years, name="year"))
    annual_count.attrs = {
        "long_name": f"Annual Water-Stressed Months (P < {threshold}*PET)",
        "units":     "months/year",
    }
    return annual_count


def compute_all_water_stress(
    pr_daily: xr.DataArray,
    tmax_daily: xr.DataArray,
    tmin_daily: xr.DataArray,
) -> xr.Dataset:
    lat_vals = pr_daily.lat.values
    doy      = _get_doy(pr_daily)

    logger.info("  Computing PET (Hargreaves-Samani)...")
    pet_daily = hargreaves_pet(tmax_daily, tmin_daily, lat_vals, doy)

    logger.info("  Computing monthly aggregates...")
    pr_monthly  = _resample_monthly(pr_daily,  "sum")
    pet_monthly = _resample_monthly(pet_daily, "sum")

    logger.info("  Computing Aridity Index...")
    ai = compute_aridity_index(pr_daily, pet_daily)

    logger.info("  Computing Moisture Deficit...")
    deficit = compute_moisture_deficit(pr_daily, pet_daily)

    logger.info("  Computing SPI-12...")
    spi12 = compute_spi(pr_monthly, scale=12)

    logger.info("  Computing Water Stress Months...")
    ws_months = compute_water_stress_months(pr_monthly, pet_monthly)

    pet_annual = _resample_yearly(pet_daily, "sum")

    ds = xr.Dataset({
        "PET_annual":        pet_annual,
        "AI":                ai,
        "MoistureDeficit":   deficit,
        "WaterStressMonths": ws_months,
        "SPI12":             spi12,     # monthly resolution kept
    })

    ds.attrs = {
        "title":      "Water Stress Metrics — Jakarta Greater Capital Region",
        "model":      MODEL,
        "PET_method": "Hargreaves-Samani",
    }
    return ds


# ===== Load helpers ===========================================================

def _open_nc(fpath: Path) -> xr.Dataset:
    return xr.open_dataset(fpath, decode_times=CFDatetimeCoder(use_cftime=True))


def _find_temp_file(scen: str, var: str, processed_dir: Path) -> Path | None:
    pattern = f"{var}_day_{MODEL}_{scen}_{ENSEMBLE}_*_jakarta.nc"
    files   = sorted(processed_dir.glob(pattern))
    if files:
        return files[0]
    # Also try without the _jakarta suffix
    pattern2 = f"{var}_day_{MODEL}_{scen}_{ENSEMBLE}_*.nc"
    files2   = sorted(processed_dir.glob(pattern2))
    return files2[0] if files2 else None


# ===== CLI ====================================================================

@click.command()
@click.option(
    "--bc-dir",
    default=str(DEFAULT_BC_DIR),
    show_default=True,
    help="Directory containing bias-corrected pr files (py/esgf/bias_corrected/).",
)
@click.option(
    "--processed-dir",
    default=str(DEFAULT_PROCESSED_DIR),
    show_default=True,
    help="Directory containing processed tasmax/tasmin files (py/esgf/processed/).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save water stress output files.",
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
def main(bc_dir, processed_dir, output_dir, scenario, method):
    bc_path        = Path(bc_dir)
    processed_path = Path(processed_dir)
    out_path       = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not bc_path.exists():
        logger.error(f"Bias-corrected directory not found: {bc_path}")
        logger.error("Run quantile_mapping.py apply --scenario all first.")
        sys.exit(1)

    scenarios_to_run = SCENARIOS if scenario == "all" else [scenario]

    completed = []
    failed    = []

    for scen in scenarios_to_run:
        logger.info(f"{'='*44}")
        logger.info(f"Processing: {scen}")
        logger.info(f"{'='*44}")

        # Load precipitation (bias-corrected)
        if scen == "historical":
            pr_fname = f"pr_day_{MODEL}_historical_{ENSEMBLE}_jakarta_bc_input.nc"
        else:
            pr_fname = f"pr_day_{MODEL}_{scen}_{ENSEMBLE}_jakarta_bc_{method}.nc"

        pr_path = bc_path / pr_fname
        if not pr_path.exists():
            logger.warning(f"  pr file not found: {pr_path} — skipping {scen}")
            failed.append(scen)
            continue

        logger.info(f"  Loading pr : {pr_fname}")
        ds_pr = _open_nc(pr_path)
        pr    = ds_pr["pr"] if "pr" in ds_pr else ds_pr["precip"]
        if pr.attrs.get("units", "") in ["kg m-2 s-1", "kg/m2/s"]:
            pr = pr * 86400.0
            pr.attrs["units"] = "mm/day"

        logger.info(
            f"  pr period  : {str(ds_pr.time.values[0])[:10]} -> "
            f"{str(ds_pr.time.values[-1])[:10]}  |  {len(ds_pr.time):,} days"
        )

        # Load temperature
        tmax_path = _find_temp_file(scen, "tasmax", processed_path)
        tmin_path = _find_temp_file(scen, "tasmin", processed_path)

        if tmax_path and tmin_path:
            logger.info(f"  Loading tasmax: {tmax_path.name}")
            logger.info(f"  Loading tasmin: {tmin_path.name}")
            ds_tmax = _open_nc(tmax_path)
            ds_tmin = _open_nc(tmin_path)
            tmax    = ds_tmax["tasmax"]
            tmin    = ds_tmin["tasmin"]

            # Convert K → °C if needed
            if tmax.attrs.get("units", "") == "K" or float(tmax.isel(time=0).mean()) > 200:
                logger.info("  Converting temperature: K -> °C")
                tmax = tmax - 273.15
                tmin = tmin - 273.15
                tmax.attrs["units"] = "degC"
                tmin.attrs["units"] = "degC"
            try:
                pr, tmax, tmin = xr.align(pr, tmax, tmin, join="inner", copy=False)
                if len(pr.time) == 0:
                    raise ValueError("No overlapping time steps after alignment.")
                logger.info(f"  Aligned to {len(pr.time):,} common time steps")
            except Exception as align_err:
                logger.warning(
                    f"  Time alignment failed ({align_err}). "
                    "Falling back to climatological temperature range."
                )
                tmax_path = None

        if not tmax_path or not tmin_path:
            # Fallback: climatological Tmax/Tmin for Jakarta
            logger.warning(
                "  tasmax/tasmin not found in processed/ — "
                "using climatological fallback (Tmax=32°C, Tmin=24°C)"
            )
            ones  = xr.ones_like(pr)
            tmax  = ones * 32.0
            tmin  = ones * 24.0
            tmax.attrs["units"] = "degC"
            tmin.attrs["units"] = "degC"

        try:
            ds_out = compute_all_water_stress(pr, tmax, tmin)
            ds_out.attrs["scenario"] = scen
        except Exception as e:
            logger.error(f"  Failed to compute water stress for {scen}: {e}")
            failed.append(scen)
            ds_pr.close()
            continue

        # Save output
        out_file = out_path / f"{MODEL}_{scen}_water_stress_jakarta.nc"
        ds_out.to_netcdf(
            out_file,
            encoding={v: {"dtype": "float32"} for v in ds_out.data_vars},
        )
        logger.info(f"  Saved: {out_file.name}")

        # Quick summary
        for var in ["AI", "MoistureDeficit", "WaterStressMonths"]:
            if var in ds_out:
                vals   = ds_out[var].values.astype(float)
                finite = vals[np.isfinite(vals)]
                if len(finite) > 0:
                    logger.info(
                        f"  {var:<20}: mean={float(finite.mean()):.2f}  "
                        f"[{ds_out[var].attrs.get('units', '')}]"
                    )

        ds_pr.close()
        completed.append(scen)

    # ===== Final summary ====================
    logger.info(f"{'='*44}")
    logger.info(f"DONE  —  {len(completed)} scenario(s) processed")
    logger.info(f"{'='*44}")
    for scen in completed:
        out = out_path / f"{MODEL}_{scen}_water_stress_jakarta.nc"
        logger.info(f"  {scen:<12} -> {out.name}")
    if failed:
        logger.warning(f"  Skipped : {failed}")
    logger.info(f"\nOutput directory: {out_path}")


if __name__ == "__main__":
    """
    References:
    Sahin, S. (2012). An aridity index defined by precipitation and specific humidity. Journal of Hydrology, 444-445, pp. 199-208. doi:10.1016/j.jhydrol.2012.04.019
    """
    main()