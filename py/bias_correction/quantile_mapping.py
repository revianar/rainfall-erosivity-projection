import sys
import numpy as np
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d
from pathlib import Path
import pickle
import click
import logging
from typing import Literal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHIRPS_DIR    = PROJECT_ROOT / "py" / "esgf"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "py" / "esgf" / "processed"
DEFAULT_OUTPUT_DIR    = PROJECT_ROOT / "py" / "esgf" / "bias_corrected"

JAKARTA_BBOX = {
    "lat_min": -8.75,
    "lat_max": -3.75,
    "lon_min": 103.125,
    "lon_max": 111.875,
}

MODEL        = "HadGEM2-AO"
ENSEMBLE     = "r1i1p1"
SCENARIOS    = ["rcp26", "rcp45", "rcp85"]
CHIRPS_START = 1981
CHIRPS_END   = 2005

WET_DAY_THRESHOLD = 1.0   # mm/day
N_QUANTILES       = 100   # Number of quantile bins

# How many days to write per chunk when streaming to disk.
# 365 days = one year at a time — keeps RAM usage low.
CHUNK_DAYS = 365


# ===== Empirical QM ====================

def fit_eqm_transfer(
    obs: np.ndarray,
    hist: np.ndarray,
    n_quantiles: int = N_QUANTILES,
    threshold: float = WET_DAY_THRESHOLD,
) -> dict:
    quantiles = np.linspace(0, 1, n_quantiles + 1)

    obs_wet  = obs[obs >= threshold]
    hist_wet = hist[hist >= threshold]

    obs_cdf  = np.quantile(obs_wet,  quantiles) if len(obs_wet)  > 0 else np.zeros(len(quantiles))
    hist_cdf = np.quantile(hist_wet, quantiles) if len(hist_wet) > 0 else np.zeros(len(quantiles))

    return {
        "method":        "empirical",
        "quantiles":     quantiles,
        "obs_cdf":       obs_cdf,
        "hist_cdf":      hist_cdf,
        "wet_prob_obs":  len(obs_wet)  / max(len(obs[~np.isnan(obs)]),  1),
        "wet_prob_hist": len(hist_wet) / max(len(hist[~np.isnan(hist)]), 1),
        "threshold":     threshold,
    }


def apply_eqm(
    data: np.ndarray,
    transfer: dict,
    preserve_dry_days: bool = True,
) -> np.ndarray:
    corrected = data.copy()
    threshold = transfer["threshold"]
    wet_mask  = data >= threshold

    if preserve_dry_days:
        prob_ratio = transfer["wet_prob_obs"] / max(transfer["wet_prob_hist"], 1e-6)
        if prob_ratio < 1.0:
            wet_indices = np.where(wet_mask)[0]
            n_to_dry    = int(len(wet_indices) * (1 - prob_ratio))
            if n_to_dry > 0:
                wet_precip = data[wet_indices]
                to_dry     = wet_indices[np.argsort(wet_precip)[:n_to_dry]]
                corrected[to_dry] = 0.0
                wet_mask[to_dry]  = False

    if wet_mask.sum() > 0:
        wet_data = data[wet_mask]
        f_interp = interp1d(
            transfer["hist_cdf"], transfer["obs_cdf"],
            bounds_error=False,
            fill_value=(transfer["obs_cdf"][0], transfer["obs_cdf"][-1])
        )
        corrected[wet_mask] = f_interp(wet_data)
        corrected[wet_mask] = np.maximum(corrected[wet_mask], threshold)

    corrected[~wet_mask & (corrected > 0)] = 0.0
    corrected = np.maximum(corrected, 0.0)
    return corrected


# ===== Parametric QM (Gamma) ====================

def fit_pqm_transfer(
    obs: np.ndarray,
    hist: np.ndarray,
    threshold: float = WET_DAY_THRESHOLD,
) -> dict:
    obs_wet  = obs[obs >= threshold]
    hist_wet = hist[hist >= threshold]

    obs_params  = stats.gamma.fit(obs_wet,  floc=0) if len(obs_wet)  > 10 else (1, 0, 1)
    hist_params = stats.gamma.fit(hist_wet, floc=0) if len(hist_wet) > 10 else (1, 0, 1)

    return {
        "method":        "parametric_gamma",
        "obs_gamma":     obs_params,
        "hist_gamma":    hist_params,
        "wet_prob_obs":  len(obs_wet)  / max(len(obs[~np.isnan(obs)]),  1),
        "wet_prob_hist": len(hist_wet) / max(len(hist[~np.isnan(hist)]), 1),
        "threshold":     threshold,
    }


def apply_pqm(data: np.ndarray, transfer: dict) -> np.ndarray:
    corrected = data.copy()
    threshold = transfer["threshold"]
    wet_mask  = data >= threshold

    if wet_mask.sum() > 0:
        wet_data   = data[wet_mask]
        obs_shape,  obs_loc,  obs_scale  = transfer["obs_gamma"]
        hist_shape, hist_loc, hist_scale = transfer["hist_gamma"]

        p = stats.gamma.cdf(wet_data, hist_shape, loc=hist_loc, scale=hist_scale)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        corrected[wet_mask] = stats.gamma.ppf(p, obs_shape, loc=obs_loc, scale=obs_scale)
        corrected[wet_mask] = np.maximum(corrected[wet_mask], threshold)

    corrected[~wet_mask] = 0.0
    return corrected


# ===== Spatial Application ====================

def apply_bias_correction_spatial(
    obs_ds: xr.Dataset,
    hist_ds: xr.Dataset,
    future_ds: xr.Dataset,
    method: Literal["empirical", "parametric"] = "empirical",
    pr_var: str = "pr",
) -> xr.DataArray:
    # Regrid obs (CHIRPS 0.25°) and hist to the model grid using nearest-neighbour.
    # The reason is because there were lots of NaNs after interpolation. 😭
    model_lat = future_ds[pr_var].lat
    model_lon = future_ds[pr_var].lon
    obs_regrid  = obs_ds[pr_var].interp(lat=model_lat, lon=model_lon, method="nearest")
    hist_regrid = hist_ds[pr_var].interp(lat=model_lat, lon=model_lon, method="nearest")

    # Fill ocean cells (NaN in CHIRPS) by propagating the nearest land cell's values so that all model grid cells get a valid transfer function.
    # I thought it will just be necessary because HadGEM2-AO at 1.25°x1.875° has many ocean cells over the Java domain that have no CHIRPS observational coverage.
    obs_vals  = obs_regrid.values.copy()   # (time, nlat, nlon)
    hist_vals = hist_regrid.values.copy()

    nlat_m = len(model_lat)
    nlon_m = len(model_lon)
    import numpy as _np

    # Build list of land cell indices (cells with valid CHIRPS data)
    land_cells = []
    for _i in range(nlat_m):
        for _j in range(nlon_m):
            if _np.isfinite(obs_vals[:, _i, _j]).sum() > 0:
                land_cells.append((_i, _j))

    if len(land_cells) == 0:
        logger.error("No land cells found in CHIRPS after regridding — check domain bbox.")
    else:
        # For every ocean cell, copy values from the nearest land cell
        for _i in range(nlat_m):
            for _j in range(nlon_m):
                if _np.isfinite(obs_vals[:, _i, _j]).sum() == 0:
                    # Find nearest land cell by Euclidean distance in grid indices
                    nearest = min(land_cells,
                                  key=lambda c: (c[0]-_i)**2 + (c[1]-_j)**2)
                    obs_vals[:,  _i, _j] = obs_vals[:,  nearest[0], nearest[1]]
                    hist_vals[:, _i, _j] = hist_vals[:, nearest[0], nearest[1]]
        logger.info(f"  Ocean fill: {25 - len(land_cells)} cells filled from nearest land cell")

    obs_pr    = obs_vals
    hist_pr   = hist_vals
    future_pr = future_ds[pr_var].values

    corrected  = np.full_like(future_pr, np.nan)
    ntime, nlat, nlon = future_pr.shape
    transfer_functions = {}

    logger.info(f"Applying {method} QM over {nlat}x{nlon} grid cells...")

    for i in range(nlat):
        for j in range(nlon):
            obs_1d    = obs_pr[:, i, j]
            hist_1d   = hist_pr[:, i, j]
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
            "reference_obs":   "CHIRPS v2.0",
            "units":           "mm/day",
        },
    )
    return da_corrected, transfer_functions


# ===== NetCDF4 append (handles leap years & partial chunks) ====================

import netCDF4 as nc4

def _nc4_append(out_path: Path, data: np.ndarray, times, var_name: str) -> None:
    with nc4.Dataset(out_path, "a") as nc_out:
        t_dim  = nc_out.variables["time"]
        t_start = len(t_dim)                      # current length before append
        t_end   = t_start + len(times)
        calendar = getattr(t_dim, "calendar", "standard")
        units    = t_dim.units

        if hasattr(times[0], "year"):
            t_num = nc4.date2num(list(times), units=units, calendar=calendar)
        else:
            import pandas as pd
            py_times = pd.DatetimeIndex(times).to_pydatetime().tolist()
            t_num = nc4.date2num(py_times, units=units, calendar=calendar)

        t_dim[t_start:t_end] = t_num

        # Append data values
        v = nc_out.variables[var_name]
        if data.ndim == 1:
            v[t_start:t_end] = data
        elif data.ndim == 3:
            v[t_start:t_end, :, :] = data
        else:
            v[t_start:t_end, ...] = data


# ===== CHIRPS processing ====================

def crop_chirps_year(nc_path: Path, bbox: dict = JAKARTA_BBOX) -> xr.Dataset:
    ds = xr.open_dataset(
        nc_path,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    if "precip" not in ds:
        raise ValueError(
            f"Expected variable 'precip' in {nc_path.name}, "
            f"found: {list(ds.data_vars)}"
        )

    ds["precip"] = ds["precip"].where(ds["precip"] >= 0)

    ds_cropped = ds.sel(
        latitude=slice(bbox["lat_min"],  bbox["lat_max"]),
        longitude=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    ds_cropped = ds_cropped.rename({"latitude": "lat", "longitude": "lon"})
    return ds_cropped


def merge_chirps_streaming(
    chirps_dir: Path,
    out_path: Path,
    start_year: int,
    end_year: int,
) -> None:
    logger.info(f"{'='*42}")
    logger.info(f"STEP 1: Merging CHIRPS files ({start_year}-{end_year} one year at a time)")
    logger.info(f"{'='*42}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()
        logger.info(f"  Removed existing file: {out_path.name}")

    missing     = []
    first_write = True
    total_days  = 0
    pr_sum      = 0.0
    pr_max_all  = 0.0

    global_attrs = {
        "title":      "CHIRPS v2.0 Daily Precipitation - Jakarta Greater Capital Region",
        "source":     "Climate Hazards Group InfraRed Precipitation with Station data (CHIRPS v2.0)",
        "resolution": "0.25 degrees",
        "domain":     "Jakarta (Jabodetabek): 5.5S-7.0S, 106E-107.5E",
        "period":     f"{start_year}-{end_year}",
        "processing": "Merged from annual files; fill values masked; coords renamed to lat/lon",
    }

    for year in range(start_year, end_year + 1):
        fname = f"chirps-v2.0.{year}.days_p25.nc"
        fpath = chirps_dir / fname

        if not fpath.exists():
            logger.warning(f"  Missing: {fname}")
            missing.append(year)
            continue

        try:
            ds_year = crop_chirps_year(fpath)
            ds_year["precip"].attrs.update({
                "units":     "mm/day",
                "long_name": "Daily Precipitation",
            })

            if first_write:
                # First year: use xarray to create the file with all metadata
                ds_year.attrs.update(global_attrs)
                ds_year.to_netcdf(
                    out_path,
                    mode="w",
                    unlimited_dims=["time"],
                    encoding={"precip": {"dtype": "float32", "zlib": True, "complevel": 4}},
                )
                first_write = False
            else:
                # Subsequent years: append with netCDF4 directly (handles any n_days)
                _nc4_append(
                    out_path,
                    data     = ds_year["precip"].values,
                    times    = ds_year["time"].values,
                    var_name = "precip",
                )

            vals   = ds_year["precip"].values.astype(float)
            finite = vals[np.isfinite(vals)]
            total_days += len(ds_year.time)
            if len(finite) > 0:
                pr_sum    += finite.sum()
                pr_max_all = max(pr_max_all, finite.max())

            logger.info(
                f"  {year} success  ({len(ds_year.time)} days | "
                f"lat [{float(ds_year.lat.min()):.2f}, {float(ds_year.lat.max()):.2f}] | "
                f"lon [{float(ds_year.lon.min()):.2f}, {float(ds_year.lon.max()):.2f}])"
            )

            ds_year.close()
            del ds_year, vals, finite

        except Exception as e:
            logger.error(f"  Failed to process {fname}: {e}")

    if first_write:
        logger.error("No CHIRPS files were written. Check --chirps-dir path.")
        sys.exit(1)

    if missing:
        logger.warning(f"  Missing years: {missing}")

    pr_mean_all = pr_sum / total_days if total_days > 0 else float("nan")
    logger.info(f"\n  Saved [CHIRPS]: {out_path.name}")
    logger.info(f"    Days   : {total_days:,}")
    logger.info(f"    Mean   : {pr_mean_all:.2f} mm/day")
    logger.info(f"    Max    : {pr_max_all:.2f} mm/day")


# ===== Load model files ====================

def load_model_lazy(fpath: Path, label: str) -> xr.Dataset:
    logger.info(f"  Opening [{label}]: {fpath.name}")

    ds = xr.open_dataset(
        fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        chunks={"time": CHUNK_DAYS},
    )

    if ds["pr"].attrs.get("units", "") in ["kg m-2 s-1", "kg/m2/s"]:
        logger.info(f"    Converting units: kg m-2 s-1 -> mm/day")
        ds["pr"] = ds["pr"] * 86400.0
        ds["pr"].attrs["units"] = "mm/day"

    logger.info(
        f"    Period : {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}"
        f"  |  {len(ds.time):,} days  |  {len(ds.lat)} lat x {len(ds.lon)} lon"
    )
    return ds


def save_model_streaming(ds: xr.Dataset, out_path: Path, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        out_path.unlink()

    n_days     = len(ds.time)
    n_chunks   = int(np.ceil(n_days / CHUNK_DAYS))
    pr_sum     = 0.0
    pr_max_all = 0.0

    t_start_str = str(ds.time.values[0])[:10]
    t_end_str   = str(ds.time.values[-1])[:10]

    logger.info(f"  Writing [{label}] in {n_chunks} chunk(s) of up to {CHUNK_DAYS} days...")

    for chunk_idx in range(n_chunks):
        c_start  = chunk_idx * CHUNK_DAYS
        c_end    = min(c_start + CHUNK_DAYS, n_days)
        ds_chunk = ds.isel(time=slice(c_start, c_end)).load()

        if chunk_idx == 0:
            # First chunk: xarray creates the file with full metadata
            ds_chunk.to_netcdf(
                out_path,
                mode="w",
                unlimited_dims=["time"],
                encoding={"pr": {"dtype": "float32", "zlib": True, "complevel": 4}},
            )
        else:
            # All other chunks (including final partial): append with netCDF4
            _nc4_append(
                out_path,
                data     = ds_chunk["pr"].values,
                times    = ds_chunk["time"].values,
                var_name = "pr",
            )

        raw    = ds_chunk["pr"].values.astype(float)
        finite = raw[np.isfinite(raw)]
        if len(finite) > 0:
            pr_sum    += finite.sum()
            pr_max_all = max(pr_max_all, finite.max())

        del ds_chunk, raw, finite

        if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
            logger.info(f"    Chunk {chunk_idx+1}/{n_chunks} written")

    ds.close()

    pr_mean_all = pr_sum / n_days if n_days > 0 else float("nan")

    logger.info(f"\n  Saved [{label}]: {out_path.name}")
    logger.info(f"    Period : {t_start_str} -> {t_end_str}")
    logger.info(f"    Days   : {n_days:,}")
    logger.info(f"    Mean   : {pr_mean_all:.2f} mm/day")
    logger.info(f"    Max    : {pr_max_all:.2f} mm/day")


def load_historical(processed_dir: Path) -> xr.Dataset:
    logger.info(f"{'='*42}")
    logger.info("STEP 2: Loading historical model file")
    logger.info(f"{'='*42}")

    pattern = f"pr_day_{MODEL}_historical_{ENSEMBLE}_*_jakarta.nc"
    files   = sorted(processed_dir.glob(pattern))

    if not files:
        logger.error(f"No historical file found in: {processed_dir}")
        logger.error(f"Expected pattern: {pattern}")
        sys.exit(1)

    if len(files) > 1:
        logger.warning(f"Multiple historical files found - using: {files[0].name}")

    return load_model_lazy(files[0], "historical")


def load_future_scenario(processed_dir: Path, scenario: str) -> xr.Dataset:
    pattern = f"pr_day_{MODEL}_{scenario}_{ENSEMBLE}_*_jakarta.nc"
    files   = sorted(processed_dir.glob(pattern))

    if not files:
        logger.warning(f"  No file found for {scenario} - pattern: {pattern}")
        return None

    if len(files) > 1:
        logger.warning(f"  Multiple files for {scenario} - using: {files[0].name}")

    return load_model_lazy(files[0], scenario)


# ===== CLI ====================

@click.group()
def cli():
    pass

# ===== Prepare subcommands ====================

@cli.command("prepare")
@click.option(
    "--chirps-dir",
    default=str(DEFAULT_CHIRPS_DIR),
    show_default=True,
    help="Directory containing annual CHIRPS .nc files.",
)
@click.option(
    "--processed-dir",
    default=str(DEFAULT_PROCESSED_DIR),
    show_default=True,
    help="Directory containing processed (Jakarta-cropped) CMIP5 model files.",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save prepared files.",
)
@click.option("--chirps-start", default=CHIRPS_START, show_default=True)
@click.option("--chirps-end",   default=CHIRPS_END,   show_default=True)
def prepare(chirps_dir, processed_dir, output_dir, chirps_start, chirps_end):
    """Merge CHIRPS files and copy model files into bias_corrected/."""
    chirps_path    = Path(chirps_dir)
    processed_path = Path(processed_dir)
    output_path    = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root  : {PROJECT_ROOT}")
    logger.info(f"CHIRPS dir    : {chirps_path}")
    logger.info(f"Processed dir : {processed_path}")
    logger.info(f"Output dir    : {output_path}")

    chirps_out = output_path / f"chirps_v2_jakarta_{chirps_start}_{chirps_end}.nc"
    merge_chirps_streaming(chirps_path, chirps_out, chirps_start, chirps_end)

    ds_hist  = load_historical(processed_path)
    hist_out = output_path / f"pr_day_{MODEL}_historical_{ENSEMBLE}_jakarta_bc_input.nc"
    save_model_streaming(ds_hist, hist_out, "historical")

    logger.info(f"{'='*42}")
    logger.info("Loading future RCP scenario files")
    logger.info(f"{'='*42}")

    for scen in SCENARIOS:
        ds_future = load_future_scenario(processed_path, scen)
        if ds_future is None:
            continue
        future_out = output_path / f"pr_day_{MODEL}_{scen}_{ENSEMBLE}_jakarta_bc_input.nc"
        save_model_streaming(ds_future, future_out, scen)

    logger.info(f"{'='*42}")
    logger.info("ALL DONE - Files ready for bias correction:")
    logger.info(f"{'='*42}")
    for f in sorted(output_path.glob("*.nc")):
        logger.info(f"  {f.name}")
    logger.info(
        f"\nRun bias correction with:\n"
        f"  python quantile_mapping.py apply --scenario rcp26\n"
        f"  python quantile_mapping.py apply --scenario rcp45\n"
        f"  python quantile_mapping.py apply --scenario rcp85\n"
        f"  python quantile_mapping.py apply --scenario all"
    )


# ===== Apply: core logic ====================

def _run_one_scenario(
    scenario: str,
    obs_path: Path,
    hist_path: Path,
    output_path: Path,
    method: str,
    calib_start: int,
    calib_end: int,
    save_transfer: bool,
) -> bool:
    """
    Run QM bias correction for a single scenario.
    Returns True on success, False if the future input file is missing.
    Obs and hist datasets are loaded inside this function so they are
    released from RAM after each scenario when running --scenario all.
    """
    future_path = Path(
        DEFAULT_OUTPUT_DIR / f"pr_day_{MODEL}_{scenario}_{ENSEMBLE}_jakarta_bc_input.nc"
    )

    for p, label in [(obs_path, "obs"), (hist_path, "hist"), (future_path, "future")]:
        if not p.exists():
            logger.error(f"File not found [{label}]: {p}")
            logger.error("Run the 'prepare' subcommand first.")
            return False

    logger.info(f"{'='*42}")
    logger.info(f"APPLY: Bias correction for {scenario.upper()}")
    logger.info(f"  Method     : {method} QM")
    logger.info(f"  Calib      : {calib_start}-{calib_end}")
    logger.info(f"  Obs        : {obs_path.name}")
    logger.info(f"  Historical : {hist_path.name}")
    logger.info(f"  Future     : {future_path.name}")
    logger.info(f"{'='*42}")

    # Load obs (CHIRPS)
    logger.info("Loading CHIRPS obs...")
    ds_obs_full = xr.open_dataset(obs_path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    if "precip" in ds_obs_full and "pr" not in ds_obs_full:
        ds_obs_full = ds_obs_full.rename({"precip": "pr"})

    # Load historical model
    logger.info("Loading historical model...")
    ds_hist_full = xr.open_dataset(hist_path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))

    # Slice to calibration period
    logger.info(f"Slicing to calibration period {calib_start}-{calib_end}...")
    obs_years  = ds_obs_full.time.dt.year.values
    hist_years = ds_hist_full.time.dt.year.values

    ds_obs_cal  = ds_obs_full.isel(
        time=np.where((obs_years  >= calib_start) & (obs_years  <= calib_end))[0]
    )
    ds_hist_cal = ds_hist_full.isel(
        time=np.where((hist_years >= calib_start) & (hist_years <= calib_end))[0]
    )
    logger.info(f"  Obs  calibration days : {len(ds_obs_cal.time):,}")
    logger.info(f"  Hist calibration days : {len(ds_hist_cal.time):,}")

    # Load future
    logger.info(f"Loading future scenario ({scenario})...")
    ds_future = xr.open_dataset(future_path, decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
    logger.info(f"  Future days: {len(ds_future.time):,}")

    # Run QM
    logger.info(f"\nRunning {method} QM...")
    da_corrected, transfers = apply_bias_correction_spatial(
        obs_ds    = ds_obs_cal,
        hist_ds   = ds_hist_cal,
        future_ds = ds_future,
        method    = method,
        pr_var    = "pr",
    )

    # Save output
    out_file = output_path / f"pr_day_{MODEL}_{scenario}_{ENSEMBLE}_jakarta_bc_{method}.nc"
    ds_out   = da_corrected.to_dataset(name="pr")
    ds_out.attrs.update({
        "title":           f"Bias-corrected HadGEM2-AO {scenario} precipitation",
        "bias_correction": f"Quantile Mapping ({method})",
        "reference_obs":   "CHIRPS v2.0",
        "calib_period":    f"{calib_start}-{calib_end}",
        "model":           MODEL,
        "scenario":        scenario,
        "ensemble":        ENSEMBLE,
    })
    logger.info("\nSaving corrected output...")
    save_model_streaming(ds_out, out_file, f"{scenario} corrected")

    # Optionally save transfer functions
    if save_transfer:
        tf_file = output_path / f"transfer_functions_{scenario}_{method}.pkl"
        with open(tf_file, "wb") as f_pkl:
            pickle.dump(transfers, f_pkl)
        logger.info(f"  Transfer functions saved: {tf_file.name}")

    # Summary
    raw_mean  = float(ds_future["pr"].values[np.isfinite(ds_future["pr"].values)].mean())
    corr_vals = da_corrected.values
    corr_mean = float(corr_vals[np.isfinite(corr_vals)].mean())

    logger.info(f"{'='*42}")
    logger.info(f"DONE: {out_file.name}")
    logger.info(f"  Raw future mean  : {raw_mean:.2f} mm/day")
    logger.info(f"  Corrected mean   : {corr_mean:.2f} mm/day")
    logger.info(f"  Output           : {out_file}")
    logger.info(f"{'='*42}")

    ds_obs_full.close()
    ds_hist_full.close()
    ds_future.close()

    return True


# ===== Apply subcommand ====================

@cli.command("apply")
@click.option(
    "--obs",
    default=str(DEFAULT_OUTPUT_DIR / f"chirps_v2_jakarta_{CHIRPS_START}_{CHIRPS_END}.nc"),
    show_default=True,
    help="Path to merged CHIRPS reference file.",
)
@click.option(
    "--hist",
    default=str(DEFAULT_OUTPUT_DIR / f"pr_day_{MODEL}_historical_{ENSEMBLE}_jakarta_bc_input.nc"),
    show_default=True,
    help="Path to historical model bc_input file.",
)
@click.option(
    "--scenario",
    required=True,
    type=click.Choice(SCENARIOS + ["all"]),
    help="RCP scenario to correct (rcp26 / rcp45 / rcp85 / all).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save bias-corrected output.",
)
@click.option(
    "--method",
    default="empirical",
    type=click.Choice(["empirical", "parametric"]),
    show_default=True,
    help="QM method: empirical (eQM) or parametric Gamma (pQM).",
)
@click.option(
    "--calib-start", default=1981, show_default=True,
    help="First year of calibration period.",
)
@click.option(
    "--calib-end", default=2005, show_default=True,
    help="Last year of calibration period.",
)
@click.option(
    "--save-transfer", is_flag=True, default=False,
    help="Save transfer functions as a .pkl file for inspection.",
)
def apply(obs, hist, scenario, output_dir, method,
          calib_start, calib_end, save_transfer):
    """
    Run quantile mapping bias correction.

    Use --scenario all to process rcp26, rcp45, and rcp85 in one go.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    obs_path  = Path(obs)
    hist_path = Path(hist)

    scenarios_to_run = SCENARIOS if scenario == "all" else [scenario]

    if scenario == "all":
        logger.info(f"{'='*42}")
        logger.info("Running ALL scenarios: rcp26, rcp45, rcp85")
        logger.info(f"{'='*42}")

    completed = []
    failed    = []

    for scen in scenarios_to_run:
        success = _run_one_scenario(
            scenario     = scen,
            obs_path     = obs_path,
            hist_path    = hist_path,
            output_path  = output_path,
            method       = method,
            calib_start  = calib_start,
            calib_end    = calib_end,
            save_transfer= save_transfer,
        )
        if success:
            completed.append(scen)
        else:
            failed.append(scen)

    if len(scenarios_to_run) > 1:
        logger.info(f"{'='*42}")
        logger.info("ALL SCENARIOS COMPLETE")
        logger.info(f"{'='*42}")
        for scen in completed:
            out = output_path / f"pr_day_{MODEL}_{scen}_{ENSEMBLE}_jakarta_bc_{method}.nc"
            logger.info(f"  {scen} -> {out.name}")
        if failed:
            logger.warning(f"  Failed: {failed}")


if __name__ == "__main__":
    cli()