import sys
import xarray as xr
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional
import pandas as pd
import click
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BC_DIR        = PROJECT_ROOT / "py" / "data" / "bias_corrected"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "py" / "data" / "processed"
DEFAULT_OUTPUT_DIR    = PROJECT_ROOT / "py" / "results" / "water_stress"

MODELS = {
    "MRI-ESM2-0": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gn",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
    "EC-Earth3": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gr",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
    "CNRM-CM6-1": {
        "ensemble":  "r1i1p1f2",
        "grid":      "gr",
        "scenarios": ["historical", "ssp126", "ssp245", "ssp585"],
    },
}

ALL_SCENARIOS = ["historical", "ssp126", "ssp245", "ssp585"]

HIST_PERIOD = (1950, 2014)
NEAR_PERIOD = (2021, 2050)
FAR_PERIOD  = (2071, 2100)

# ===== Scenario-aware temperature warming deltas ==============================
#
# Source: IPCC AR6 WGI, Atlas Chapter & Chapter 4 (Table 4.5 / Interactive Atlas),
# Southeast Asia (SEA) reference region, CMIP6 multi-model ensemble, relative to 1995-2014 baseline.
#
# Two statistics are provided per scenario × period:
#   median : best estimate (50th percentile of CMIP6 ensemble)
#   range  : mean of 5th-95th percentile bounds (likely range midpoint)
#
# Periods align with NEAR_PERIOD (2021-2050 ≈ AR6 mid-term 2041-2060 midpoint)
# and FAR_PERIOD (2071-2100 ≈ AR6 long-term 2081-2100).
#
# These are applied as deltas to the observed climatological baseline:
#   Tmax_baseline = 32.0°C  (Jakarta dry-season mean Tmax, BMKG climatology)
#   Tmin_baseline = 24.0°C  (Jakarta mean Tmin, BMKG climatology)
#
# Asymmetric warming: Tmax warms 10% faster than Tmin, consistent with
# diurnal temperature range (DTR) narrowing observed in tropical land areas
# (IPCC AR6 WGI Chapter 2, Section 2.3.1.1).
#
# Reference:
#   IPCC, 2021: Atlas. In: Climate Change 2021: The Physical Science Basis.
#   Contribution of WGI to AR6. Cambridge University Press. doi:10.1017/
#   9781009157896.021. SEA region values extracted from Interactive Atlas.
#
WARMING_DELTA = {
    #          near-term (2021-2050)       far-term (2071-2100)
    #          median   range_mid          median   range_mid
    "ssp126": {"near": {"median": 0.8, "range_mid": 0.9},
               "far":  {"median": 1.0, "range_mid": 1.1}},
    "ssp245": {"near": {"median": 1.0, "range_mid": 1.1},
               "far":  {"median": 1.8, "range_mid": 2.0}},
    "ssp585": {"near": {"median": 1.3, "range_mid": 1.4},
               "far":  {"median": 3.5, "range_mid": 3.9}},
}

# Baseline climatological temperatures for Jakarta [°C]
# Source: BMKG 1981-2014 climatology, consistent with QDM calibration period
TMAX_BASELINE = 32.0
TMIN_BASELINE = 24.0

# Tmax warms proportionally faster than Tmin (DTR narrowing factor)
# Tmax_delta = warming_delta * TMAX_ASYMMETRY_FACTOR
# Tmin_delta = warming_delta * (1 - (TMAX_ASYMMETRY_FACTOR - 1))
# Such that weighted mean delta = warming_delta exactly.
TMAX_ASYMMETRY_FACTOR = 1.10   # Tmax 10% above mean warming
TMIN_ASYMMETRY_FACTOR = 0.90   # Tmin 10% below mean warming


# ===== Helpers ====================

def _get_years(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "year"):
        return np.array([v.year for v in t])
    return pd.DatetimeIndex(t).year.values


def _get_months(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "month"):
        return np.array([v.month for v in t])
    return pd.DatetimeIndex(t).month.values


def _get_doy(da: xr.DataArray) -> np.ndarray:
    t = da.time.values
    if hasattr(t[0], "year"):
        # cftime objects
        if hasattr(t[0], "calendar") or type(t[0]).__name__ == "Datetime360Day":
            return np.array([(v.month - 1) * 30 + v.day for v in t])
        else:
            return np.array([v.timetuple().tm_yday for v in t])
    return pd.DatetimeIndex(t).dayofyear.values


def _resample_yearly(da: xr.DataArray, func: str) -> xr.DataArray:
    years        = _get_years(da)
    unique_years = np.unique(years)
    slices = []
    for yr in unique_years:
        da_yr = da.isel(time=np.where(years == yr)[0])
        slices.append(da_yr.sum("time") if func == "sum" else da_yr.max("time"))
    return xr.concat(slices, dim=pd.Index(unique_years, name="year"))


def _resample_monthly(da: xr.DataArray, func: str) -> xr.DataArray:
    years  = _get_years(da)
    months = _get_months(da)

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


# ===== Synthetic seasonal temperature for baseline period ====================

def generate_temperature(pr, time, scenario, period, WARMING_DELTA,
                         TMAX_BASE=32.0, TMIN_BASE=24.0):
    
    # Seasonal cycle (sinusoidal)
    doy = time.dt.dayofyear.values
    
    # Tropical small amplitude (±1°C)
    seasonal_amp = 1.0
    
    seasonal_cycle = seasonal_amp * np.sin(2 * np.pi * (doy - 30) / 365)
    
    # Warming delta
    if scenario in WARMING_DELTA:
        years = time.dt.year.values
        
        near_mid = np.mean(period["near"])
        far_mid  = np.mean(period["far"])
        
        alpha = np.clip((years - near_mid) / (far_mid - near_mid), 0, 1)
        
        dT_near = WARMING_DELTA[scenario]["near"]["median"]
        dT_far  = WARMING_DELTA[scenario]["far"]["median"]
        
        dT = (1 - alpha) * dT_near + alpha * dT_far
    else:
        dT = 0.0
    
    # Random variability
    noise = np.random.normal(0, 0.3, size=pr.shape)
    
    # Combine components
    tmax = TMAX_BASE + seasonal_cycle + dT + noise
    tmin = TMIN_BASE + seasonal_cycle * 0.8 + dT + noise * 0.8
    
    return tmax, tmin


# ===== Scenario-aware temperature construction ====================

def _get_period_key(year: int) -> str:
    """Map a year to 'near' or 'far' warming period."""
    if year <= NEAR_PERIOD[1]:
        return "near"
    return "far"


def build_scenario_temperature(
    pr: xr.DataArray,
    scenario: str,
    stat: str = "median",
) -> tuple:
    """
    Build daily Tmax and Tmin arrays for a given scenario using IPCC AR6
    SEA region warming deltas applied to the Jakarta climatological baseline.

    The delta is applied linearly across the projection period, interpolating
    between the near-term (2021-2050) and far-term (2071-2100) midpoint values
    so that temperature increases smoothly rather than stepping discretely.

    Parameters
    ----------
    pr       : DataArray with time dimension, used only for shape/coords
    scenario : SSP scenario key (e.g. 'ssp245')
    stat     : 'median' or 'range_mid'

    Returns
    -------
    tmax, tmin : xr.DataArray pair with same coords as pr
    """
    if scenario not in WARMING_DELTA:
        logger.warning(
            f"  No warming delta defined for {scenario}. "
            "Using flat baseline (Tmax=32°C, Tmin=24°C)."
        )
        ones = xr.ones_like(pr)
        tmax = ones * TMAX_BASELINE
        tmin = ones * TMIN_BASELINE
        tmax.attrs["units"] = "degC"
        tmin.attrs["units"] = "degC"
        return tmax, tmin

    deltas    = WARMING_DELTA[scenario]
    near_val  = deltas["near"][stat]
    far_val   = deltas["far"][stat]

    # Reference years for interpolation
    near_mid = (NEAR_PERIOD[0] + NEAR_PERIOD[1]) / 2   # 2035.5
    far_mid  = (FAR_PERIOD[0]  + FAR_PERIOD[1])  / 2   # 2085.5

    # Extract years from time coordinate
    time_vals = pr.time.values
    if hasattr(time_vals[0], "year"):
        years = np.array([t.year for t in time_vals], dtype=float)
    else:
        years = pd.DatetimeIndex(time_vals).year.values.astype(float)

    # Linear interpolation of warming delta per day
    # Clamp to [near_val, far_val] so no extrapolation beyond defined periods
    alpha        = np.clip((years - near_mid) / (far_mid - near_mid), 0.0, 1.0)
    daily_delta  = near_val + alpha * (far_val - near_val)   # (ntime,)

    # Broadcast to spatial dims if present
    if pr.ndim == 3:
        nlat = pr.shape[1]
        nlon = pr.shape[2]
        daily_delta = daily_delta[:, np.newaxis, np.newaxis] * np.ones((1, nlat, nlon))
    elif pr.ndim == 2:
        nlat = pr.shape[1]
        daily_delta = daily_delta[:, np.newaxis] * np.ones((1, nlat))

    tmax_delta = daily_delta * TMAX_ASYMMETRY_FACTOR
    tmin_delta = daily_delta * TMIN_ASYMMETRY_FACTOR

    tmax_vals = TMAX_BASELINE + tmax_delta
    tmin_vals = TMIN_BASELINE + tmin_delta

    tmax = xr.DataArray(
        tmax_vals.astype(np.float32),
        coords=pr.coords,
        dims=pr.dims,
        attrs={
            "units":       "degC",
            "long_name":   "Maximum temperature (IPCC AR6 scenario-aware)",
            "source":      f"IPCC AR6 WGI Atlas SEA region, {stat}, {scenario}",
            "baseline":    f"{TMAX_BASELINE}°C (BMKG 1981-2014)",
            "near_delta":  f"+{near_val * TMAX_ASYMMETRY_FACTOR:.2f}°C",
            "far_delta":   f"+{far_val  * TMAX_ASYMMETRY_FACTOR:.2f}°C",
        },
    )
    tmin = xr.DataArray(
        tmin_vals.astype(np.float32),
        coords=pr.coords,
        dims=pr.dims,
        attrs={
            "units":       "degC",
            "long_name":   "Minimum temperature (IPCC AR6 scenario-aware)",
            "source":      f"IPCC AR6 WGI Atlas SEA region, {stat}, {scenario}",
            "baseline":    f"{TMIN_BASELINE}°C (BMKG 1981-2014)",
            "near_delta":  f"+{near_val * TMIN_ASYMMETRY_FACTOR:.2f}°C",
            "far_delta":   f"+{far_val  * TMIN_ASYMMETRY_FACTOR:.2f}°C",
        },
    )

    logger.info(
        f"  Scenario-aware temp [{scenario}, {stat}]: "
        f"Tmax {TMAX_BASELINE}→{TMAX_BASELINE + far_val * TMAX_ASYMMETRY_FACTOR:.1f}°C  "
        f"Tmin {TMIN_BASELINE}→{TMIN_BASELINE + far_val * TMIN_ASYMMETRY_FACTOR:.1f}°C "
        f"(by {int(FAR_PERIOD[1])})"
    )
    return tmax, tmin


# ===== Hargreaves-Samani PET ====================

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
    """Annual count of months where P < threshold × PET."""
    stressed     = (pr_monthly < threshold * pet_monthly).astype(float)
    # Align time coords so we can group by year
    years        = _get_years(stressed)
    unique_years = np.unique(years)
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
    model: str = "",
    scenario: str = "",
) -> xr.Dataset:
    lat_vals = pr_daily.lat.values
    doy      = _get_doy(pr_daily)      # cftime-safe day-of-year

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
        "model":      model,
        "scenario":   scenario,
        "PET_method": "Hargreaves-Samani",
    }
    return ds


# ===== Load helpers ===========================================================

def _open_nc(fpath: Path) -> xr.Dataset:
    return xr.open_dataset(fpath, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))


def _find_temp_file(scenario: str, var: str, model: str, ensemble: str, processed_dir: Path) -> Optional[Path]:
    pattern = f"{var}_day_{model}_{scenario}_{ensemble}_*_jakarta.nc"
    files   = sorted(processed_dir.glob(pattern))
    if files:
        return files[0]
    pattern2 = f"{var}_day_{model}_{scenario}_{ensemble}_*.nc"
    files2   = sorted(processed_dir.glob(pattern2))
    return files2[0] if files2 else None


# ===== CLI ====================================================================

@click.command()
@click.option(
    "--bc-dir",
    default=str(DEFAULT_BC_DIR),
    show_default=True,
    help="Directory containing QDM bias-corrected pr files.",
)
@click.option(
    "--processed-dir",
    default=str(DEFAULT_PROCESSED_DIR),
    show_default=True,
    help="Directory containing processed tasmax/tasmin files (optional).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save water stress output files.",
)
@click.option(
    "--model",
    default="all",
    type=click.Choice(list(MODELS.keys()) + ["all"]),
    show_default=True,
    help="Which CMIP6 model to process, or 'all'.",
)
@click.option(
    "--scenario",
    default="all",
    type=click.Choice(ALL_SCENARIOS + ["all"]),
    show_default=True,
    help="Scenario to process, or 'all' for every scenario.",
)
@click.option(
    "--temp-stat",
    default="median",
    type=click.Choice(["median", "range_mid"]),
    show_default=True,
    help=(
        "Which IPCC AR6 warming statistic to use for the temperature delta. "
        "'median' = best estimate; 'range_mid' = mean of 5th-95th percentile."
    ),
)
def main(bc_dir, processed_dir, output_dir, model, scenario, temp_stat):
    bc_path        = Path(bc_dir)
    processed_path = Path(processed_dir)
    out_path       = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not bc_path.exists():
        logger.error(f"Bias-corrected directory not found: {bc_path}")
        logger.error("Run QDM.py apply --model all --scenario all first.")
        sys.exit(1)

    models_to_run    = list(MODELS.keys()) if model    == "all" else [model]
    scenarios_to_run = ALL_SCENARIOS       if scenario == "all" else [scenario]

    completed = []
    skipped   = []

    for mdl in models_to_run:
        ensemble = MODELS[mdl]["ensemble"]

        for scenario in scenarios_to_run:
            if scenario not in MODELS[mdl]["scenarios"]:
                logger.info(f"\nSkipping {mdl}/{scenario} — not in model scenario list.")
                skipped.append((mdl, scenario))
                continue

            logger.info(f"{'='*55}")
            logger.info(f"Model: {mdl}  |  Scenario: {scenario}  |  Temp stat: {temp_stat}")
            logger.info(f"{'='*55}")

            # Load precipitation (QDM-corrected)
            pr_fname = f"pr_day_{mdl}_{scenario}_{ensemble}_jakarta_qdm.nc"
            pr_path  = bc_path / pr_fname

            if not pr_path.exists():
                logger.warning(f"  pr file not found: {pr_path} — skipping {mdl}/{scenario}")
                skipped.append((mdl, scenario))
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

            # Try to load tasmax/tasmin from processed files first
            tmax_path = _find_temp_file(scenario, "tasmax", mdl, ensemble, processed_path)
            tmin_path = _find_temp_file(scenario, "tasmin", mdl, ensemble, processed_path)

            if tmax_path and tmin_path:
                logger.info(f"  Loading tasmax: {tmax_path.name}")
                logger.info(f"  Loading tasmin: {tmin_path.name}")
                ds_tmax = _open_nc(tmax_path)
                ds_tmin = _open_nc(tmin_path)
                tmax    = ds_tmax["tasmax"]
                tmin    = ds_tmin["tasmin"]

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
                        "Falling back to scenario-aware temperature."
                    )
                    tmax_path = None

            if not tmax_path or not tmin_path:
                # Use IPCC AR6 scenario-aware warming delta
                logger.info(
                    f"  tasmax/tasmin not found! "
                    f"  using IPCC AR6 SEA warming delta "
                    f"({temp_stat}) for {scenario}"
                )
                tmax, tmin = build_scenario_temperature(pr, scenario, stat=temp_stat)

            # Compute water stress metrics
            try:
                ds_out = compute_all_water_stress(
                    pr, tmax, tmin, model=mdl, scenario=scenario
                )
                ds_out.attrs.update({
                    "temp_source": (
                        "IPCC AR6 WGI Atlas SEA region warming delta"
                        if not tmax_path else "CMIP6 tasmax/tasmin files"
                    ),
                    "temp_stat": temp_stat,
                    "warming_near": str(WARMING_DELTA.get(scenario, {}).get("near", {})),
                    "warming_far":  str(WARMING_DELTA.get(scenario, {}).get("far",  {})),
                })
            except Exception as e:
                logger.error(f"  Failed to compute water stress for {mdl}/{scenario}: {e}")
                skipped.append((mdl, scenario))
                ds_pr.close()
                continue

            # Save output to include temp_stat in filename for traceability
            out_file = out_path / f"{mdl}_{scenario}_{ensemble}_water_stress_{temp_stat}_jakarta.nc"
            ds_out.to_netcdf(
                out_file,
                encoding={v: {"dtype": "float32"} for v in ds_out.data_vars},
            )
            logger.info(f"  Saved: {out_file.name}")

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
            completed.append((mdl, scenario))

    # ===== Final summary ====================
    logger.info(f"{'='*55}")
    logger.info(f"DONE — {len(completed)} processed, {len(skipped)} skipped")
    logger.info(f"{'='*55}")
    for mdl, scenario in completed:
        ensemble = MODELS[mdl]["ensemble"]
        out = out_path / f"{mdl}_{scenario}_{ensemble}_water_stress_{temp_stat}_jakarta.nc"
        logger.info(f"  OK  {mdl:<15} {scenario:<12} -> {out.name}")
    if skipped:
        logger.info("Skipped:")
        for item in skipped:
            logger.info(f"  --  {item[0]:<15} {item[1]}")
    logger.info(f"\nOutput directory: {out_path}")


if __name__ == "__main__":
    """
    References:
    Sahin, S. (2012). An aridity index defined by precipitation and specific
    humidity. Journal of Hydrology, 444-445, pp. 199-208.
    doi:10.1016/j.jhydrol.2012.04.019

    IPCC, 2021: Atlas. In: Climate Change 2021: The Physical Science Basis.
    Contribution of WGI to the Sixth Assessment Report of the IPCC.
    Cambridge University Press. doi:10.1017/9781009157896.021
    """
    main()