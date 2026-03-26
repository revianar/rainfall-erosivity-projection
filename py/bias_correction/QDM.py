import sys
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from pathlib import Path
import pickle
import click
import logging
import netCDF4 as nc4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT.parent

DEFAULT_CHIRPS_DIR  = DATA_ROOT / "CHIRPS"
DEFAULT_PROC_DIR    = DATA_ROOT / "Rainfall-Erosivity" / "py" / "data" / "processed"
DEFAULT_OUTPUT_DIR  = DATA_ROOT / "Rainfall-Erosivity" / "py" / "data" / "bias_corrected"

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

CALIB_START = 1981
CALIB_END   = 2014

CHIRPS_START = 1981
CHIRPS_END   = 2014          # CMIP6 historical ends 2014

WET_THRESHOLD = 1.0          # mm/day
N_QUANTILES   = 100          # quantile bins for QDM
CHUNK_DAYS    = 365          # streaming chunk size


# ===== QDM core ====================

def fit_qdm_transfer(
    obs:        np.ndarray,
    hist:       np.ndarray,
    n_quantiles: int   = N_QUANTILES,
    threshold:   float = WET_THRESHOLD,
) -> dict:
    """
    Fit a QDM transfer function from obs (CHIRPS) and hist (model calibration).

    Returns a dict containing:
      obs_cdf   : observed quantile values  [n_quantiles+1]
      hist_cdf  : historical model quantile values  [n_quantiles+1]
      quantiles : probability levels  [n_quantiles+1]  (0 to 1)
      wet_prob_obs / wet_prob_hist : wet-day probabilities
      threshold : wet day threshold
    """
    quantiles = np.linspace(0, 1, n_quantiles + 1)

    obs_wet  = obs[ obs  >= threshold]
    hist_wet = hist[hist >= threshold]

    if len(obs_wet) < 10 or len(hist_wet) < 10:
        # If not enough wet days, return identity transfer
        fallback = np.linspace(0, max(obs.max(), hist.max(), 1), n_quantiles + 1)
        return {
            "obs_cdf":       fallback,
            "hist_cdf":      fallback,
            "quantiles":     quantiles,
            "wet_prob_obs":  len(obs_wet)  / max(len(obs[~np.isnan(obs)]),   1),
            "wet_prob_hist": len(hist_wet) / max(len(hist[~np.isnan(hist)]), 1),
            "threshold":     threshold,
        }

    obs_cdf  = np.quantile(obs_wet,  quantiles)
    hist_cdf = np.quantile(hist_wet, quantiles)

    return {
        "obs_cdf":       obs_cdf,
        "hist_cdf":      hist_cdf,
        "quantiles":     quantiles,
        "wet_prob_obs":  len(obs_wet)  / max(len(obs[~np.isnan(obs)]),   1),
        "wet_prob_hist": len(hist_wet) / max(len(hist[~np.isnan(hist)]), 1),
        "threshold":     threshold,
    }


def apply_qdm(
    future:    np.ndarray,
    transfer:  dict,
    threshold: float = WET_THRESHOLD,
) -> np.ndarray:
    """
    Apply QDM to a 1-D future time series.

    Steps per wet-day value x_f:
      1. τ   = F_hist(x_f)              find quantile in hist CDF
      2. δ   = x_f / Q_hist(τ)          multiplicative delta (climate change signal)
      3. x*  = Q_obs(τ)                 map τ onto obs CDF
      4. x_c = δ × x*                   apply delta to obs-mapped value

    Dry-day frequency is adjusted using the wet-prob ratio before intensity
    correction, ensuring that the corrected series has the same wet-day
    frequency as the observations.
    """
    corrected = future.copy().astype(float)
    obs_cdf   = transfer["obs_cdf"]
    hist_cdf  = transfer["hist_cdf"]
    quantiles = transfer["quantiles"]

    # Step 1: dry-day frequency correction
    # If the model has more wet days than obs, demote the lightest wet days to dry (zero)
    # so the wet-day frequency matches observations.
    wet_mask = corrected >= threshold
    prob_obs  = transfer["wet_prob_obs"]
    prob_hist = transfer["wet_prob_hist"]

    if prob_hist > prob_obs and wet_mask.sum() > 0:
        ratio        = prob_obs / max(prob_hist, 1e-9)
        wet_indices  = np.where(wet_mask)[0]
        n_to_dry     = int(len(wet_indices) * (1.0 - ratio))
        if n_to_dry > 0:
            # Demote the lightest wet-day values first
            sorted_wet   = wet_indices[np.argsort(corrected[wet_indices])]
            demote       = sorted_wet[:n_to_dry]
            corrected[demote] = 0.0
            wet_mask[demote]  = False

    # Step 2: QDM intensity correction on remaining wet days
    if wet_mask.sum() > 0:
        x_f = corrected[wet_mask]

        # Interpolators for hist CDF and obs CDF (both indexed by quantile)
        # f_tau   : value -> quantile level   (invert hist CDF)
        # f_obs_q : quantile level -> obs value
        f_tau   = interp1d(
            hist_cdf, quantiles,
            bounds_error=False,
            fill_value=(quantiles[0], quantiles[-1]),
        )
        f_obs_q = interp1d(
            quantiles, obs_cdf,
            bounds_error=False,
            fill_value=(obs_cdf[0], obs_cdf[-1]),
        )
        f_hist_q = interp1d(
            quantiles, hist_cdf,
            bounds_error=False,
            fill_value=(hist_cdf[0], hist_cdf[-1]),
        )

        tau        = f_tau(x_f)
        x_hist_tau = f_hist_q(tau)                 # Q_hist(τ)
        x_obs_tau  = f_obs_q(tau)                  # Q_obs(τ)

        # Multiplicative delta  (clip denominator to avoid div-by-zero)
        delta      = x_f / np.maximum(x_hist_tau, 1e-6)   # step 2

        # Cap delta at 5× to avoid runaway extrapolation in extremes
        delta      = np.clip(delta, 0.0, 5.0)

        corrected[wet_mask] = delta * x_obs_tau    # step 4

        # Enforce minimum wet-day value and non-negativity
        corrected[wet_mask] = np.maximum(corrected[wet_mask], threshold)

    corrected[~wet_mask] = 0.0
    corrected            = np.maximum(corrected, 0.0)
    return corrected


# ===== Spatial QDM ============================================================

def apply_qdm_spatial(
    obs_ds:     xr.Dataset,
    hist_ds:    xr.Dataset,
    future_ds:  xr.Dataset,
    pr_var:     str = "pr",
) -> tuple:
    model_lat = future_ds[pr_var].lat
    model_lon = future_ds[pr_var].lon

    # Regrid obs and hist to model grid
    obs_regrid  = obs_ds[pr_var].interp(lat=model_lat, lon=model_lon, method="nearest")
    hist_regrid = hist_ds[pr_var].interp(lat=model_lat, lon=model_lon, method="nearest")

    obs_vals  = obs_regrid.values.copy()    # (time, nlat, nlon)
    hist_vals = hist_regrid.values.copy()

    nlat = len(model_lat)
    nlon = len(model_lon)

    # Ocean cell filling
    land_cells = [
        (i, j)
        for i in range(nlat)
        for j in range(nlon)
        if np.isfinite(obs_vals[:, i, j]).sum() > 10
    ]

    if not land_cells:
        logger.error("No land cells found — check domain bbox or CHIRPS coverage.")
    else:
        n_ocean = 0
        for i in range(nlat):
            for j in range(nlon):
                if np.isfinite(obs_vals[:, i, j]).sum() <= 10:
                    nearest = min(land_cells,
                                  key=lambda c: (c[0] - i) ** 2 + (c[1] - j) ** 2)
                    obs_vals[:,  i, j] = obs_vals[:,  nearest[0], nearest[1]]
                    hist_vals[:, i, j] = hist_vals[:, nearest[0], nearest[1]]
                    n_ocean += 1
        if n_ocean:
            logger.info(f"  Ocean fill: {n_ocean} cell(s) filled from nearest land cell")

    future_pr = future_ds[pr_var].values          # (time, nlat, nlon)
    corrected = np.full_like(future_pr, np.nan, dtype=np.float32)
    transfers = {}

    logger.info(f"  Applying QDM over {nlat} x {nlon} grid cells ...")

    for i in range(nlat):
        for j in range(nlon):
            obs_1d    = obs_vals[:,  i, j]
            hist_1d   = hist_vals[:, i, j]
            future_1d = future_pr[:, i, j]

            if np.all(np.isnan(obs_1d)) or np.all(np.isnan(hist_1d)):
                continue

            tf = fit_qdm_transfer(obs_1d, hist_1d)
            corrected[:, i, j] = apply_qdm(future_1d, tf)
            transfers[(i, j)]  = tf

        if (i + 1) % max(1, nlat // 5) == 0:
            logger.info(f"    Row {i+1}/{nlat} done")

    da_corrected = xr.DataArray(
        corrected,
        coords=future_ds[pr_var].coords,
        dims=future_ds[pr_var].dims,
        attrs={
            **future_ds[pr_var].attrs,
            "bias_correction": "Quantile Delta Mapping (QDM)",
            "reference_obs":   "CHIRPS v2.0",
            "units":           "mm/day",
        },
    )
    return da_corrected, transfers


# ===== NetCDF4 streaming append ====================

def _nc4_append(out_path: Path, data: np.ndarray, times, var_name: str) -> None:
    with nc4.Dataset(out_path, "a") as nc_out:
        t_dim    = nc_out.variables["time"]
        t_start  = len(t_dim)
        t_end    = t_start + len(times)
        calendar = getattr(t_dim, "calendar", "standard")
        units    = t_dim.units

        if hasattr(times[0], "year"):
            t_num = nc4.date2num(list(times), units=units, calendar=calendar)
        else:
            import pandas as pd
            t_num = nc4.date2num(
                pd.DatetimeIndex(times).to_pydatetime().tolist(),
                units=units, calendar=calendar
            )

        t_dim[t_start:t_end] = t_num
        v = nc_out.variables[var_name]
        if data.ndim == 3:
            v[t_start:t_end, :, :] = data
        else:
            v[t_start:t_end, ...] = data


# ===== CHIRPS processing ======================================================

def crop_chirps_year(nc_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(
        nc_path,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    if "precip" not in ds:
        raise ValueError(f"Expected 'precip' in {nc_path.name}, found {list(ds.data_vars)}")

    # Jakarta domain — keep at native CHIRPS 0.25° for regridding later
    BBOX = {"lat_min": -9.5, "lat_max": -3.0, "lon_min": 102.0, "lon_max": 113.0}
    ds["precip"] = ds["precip"].where(ds["precip"] >= 0)
    ds = ds.sel(
        latitude=slice(BBOX["lat_min"],  BBOX["lat_max"]),
        longitude=slice(BBOX["lon_min"], BBOX["lon_max"]),
    )
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    return ds


def merge_chirps_streaming(chirps_dir: Path, out_path: Path,
                            start_year: int, end_year: int) -> None:
    logger.info("=" * 50)
    logger.info(f"Merging CHIRPS {start_year}-{end_year}")
    logger.info("=" * 50)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    missing     = []
    first_write = True
    total_days  = 0

    for year in range(start_year, end_year + 1):
        fpath = chirps_dir / f"chirps-v2.0.{year}.days_p25.nc"
        if not fpath.exists():
            logger.warning(f"  Missing: {fpath.name}")
            missing.append(year)
            continue
        try:
            ds_year = crop_chirps_year(fpath)
            ds_year["precip"].attrs.update({"units": "mm/day", "long_name": "Daily Precipitation"})

            if first_write:
                ds_year.to_netcdf(out_path, mode="w", unlimited_dims=["time"],
                                   encoding={"precip": {"dtype": "float32", "zlib": True, "complevel": 4}})
                first_write = False
            else:
                _nc4_append(out_path, ds_year["precip"].values, ds_year["time"].values, "precip")

            total_days += len(ds_year.time)
            logger.info(f"  {year}  ({len(ds_year.time)} days)")
            ds_year.close()
        except Exception as e:
            logger.error(f"  Failed {year}: {e}")

    if first_write:
        logger.error("No CHIRPS files written — check --chirps-dir.")
        sys.exit(1)
    if missing:
        logger.warning(f"  Missing years: {missing}")
    logger.info(f"  Saved: {out_path.name}  ({total_days:,} days total)")


# ===== Model file helpers ====================

def load_model_lazy(fpath: Path, label: str) -> xr.Dataset:
    logger.info(f"  Loading [{label}]: {fpath.name}")
    ds = xr.open_dataset(
        fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        chunks={"time": CHUNK_DAYS},
    )
    # Unit conversion if needed
    units = ds["pr"].attrs.get("units", "")
    if units in ["kg m-2 s-1", "kg/m2/s"]:
        logger.info("    Converting units: kg m-2 s-1 -> mm/day")
        ds["pr"] = ds["pr"] * 86400.0
        ds["pr"].attrs["units"] = "mm/day"
    logger.info(
        f"    {str(ds.time.values[0])[:10]} -> {str(ds.time.values[-1])[:10]}"
        f"  ({len(ds.time):,} days  |  {len(ds.lat)} lat x {len(ds.lon)} lon)"
    )
    return ds


def save_model_streaming(ds: xr.Dataset, out_path: Path, label: str,
                          pr_var: str = "pr") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    n_days   = len(ds.time)
    n_chunks = int(np.ceil(n_days / CHUNK_DAYS))
    pr_sum   = 0.0
    pr_max   = 0.0

    logger.info(f"  Writing [{label}] ({n_chunks} chunk(s)) -> {out_path.name}")

    for idx in range(n_chunks):
        s = idx * CHUNK_DAYS
        e = min(s + CHUNK_DAYS, n_days)
        chunk = ds.isel(time=slice(s, e)).load()

        if idx == 0:
            chunk.to_netcdf(out_path, mode="w", unlimited_dims=["time"],
                             encoding={pr_var: {"dtype": "float32", "zlib": True, "complevel": 4}})
        else:
            _nc4_append(out_path, chunk[pr_var].values, chunk["time"].values, pr_var)

        raw = chunk[pr_var].values.astype(float)
        fin = raw[np.isfinite(raw)]
        if len(fin) > 0:
            pr_sum += fin.sum()
            pr_max  = max(pr_max, fin.max())
        del chunk, raw, fin

        if (idx + 1) % 10 == 0 or idx == n_chunks - 1:
            logger.info(f"    Chunk {idx+1}/{n_chunks}")

    ds.close()
    mean = pr_sum / n_days if n_days > 0 else float("nan")
    logger.info(f"    mean={mean:.2f} mm/day  max={pr_max:.2f} mm/day")


# ===== CLI 'prepare' subcommand ====================

@click.group()
def cli():
    pass


@cli.command("prepare")
@click.option("--chirps-dir",    default=str(DEFAULT_CHIRPS_DIR), show_default=True,
              help="Directory containing annual CHIRPS .nc files.")
@click.option("--processed-dir", default=str(DEFAULT_PROC_DIR),   show_default=True,
              help="Directory containing Jakarta-cropped CMIP6 model files.")
@click.option("--output-dir",    default=str(DEFAULT_OUTPUT_DIR), show_default=True,
              help="Directory to save prepared files.")
@click.option("--model",         default="all",
              type=click.Choice(list(MODELS.keys()) + ["all"]), show_default=True,
              help="Which model to prepare, or 'all'.")
@click.option("--chirps-start",  default=CHIRPS_START, show_default=True)
@click.option("--chirps-end",    default=CHIRPS_END,   show_default=True)
def prepare(chirps_dir, processed_dir, output_dir, model, chirps_start, chirps_end):
    """
    Merge CHIRPS files and copy cropped CMIP6 model files into bias_corrected/.

    CHIRPS is merged once (shared across all models).
    Model files are copied per-model.
    """
    chirps_path = Path(chirps_dir)
    proc_path   = Path(processed_dir)
    out_path    = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 50)
    logger.info("QDM.py  prepare  (CMIP6 / QDM)")
    logger.info("=" * 50)
    logger.info(f"Project root  : {PROJECT_ROOT}")
    logger.info(f"CHIRPS dir    : {chirps_path}")
    logger.info(f"Processed dir : {proc_path}")
    logger.info(f"Output dir    : {out_path}")

    # CHIRPS (shared)
    chirps_out = out_path / f"chirps_v2_jakarta_{chirps_start}_{chirps_end}.nc"
    if chirps_out.exists():
        logger.info(f"\nCHIRPS file already exists — skipping merge: {chirps_out.name}")
    else:
        merge_chirps_streaming(chirps_path, chirps_out, chirps_start, chirps_end)

    # Model files per model
    models_to_run = list(MODELS.keys()) if model == "all" else [model]

    for mdl in models_to_run:
        cfg = MODELS[mdl]
        logger.info(f"{'=' * 50}")
        logger.info(f"Model: {mdl}  (ensemble={cfg['ensemble']}  grid={cfg['grid']})")
        logger.info(f"{'=' * 50}")

        for scenario in ["historical"] + cfg["scenarios"]:
            pattern  = f"pr_day_{mdl}_{scenario}_{cfg['ensemble']}_jakarta.nc"
            src_file = proc_path / pattern

            if not src_file.exists():
                logger.warning(f"  Not found: {pattern} — run crop_domain.py first")
                continue

            dst_file = out_path / pattern
            if dst_file.exists():
                logger.info(f"  Already in output dir: {pattern}")
                continue

            # Load, validate, and re-save into bias_corrected/ with compression
            logger.info(f"  Copying: {pattern}")
            ds = load_model_lazy(src_file, f"{mdl}/{scenario}")
            save_model_streaming(ds, dst_file, f"{mdl}/{scenario}")

    logger.info("=" * 50)
    logger.info("prepare DONE — files in bias_corrected/:")
    logger.info("=" * 50)
    for f in sorted(out_path.glob("*.nc")):
        logger.info(f"  {f.name}")  
    logger.info(
        "\nNext step:\n"
        "  python QDM.py apply --model all --scenario all"
    )


# ===== Apply: core per-model-per-scenario ====================

def _run_qdm(
    model:       str,
    scenario:    str,
    output_path: Path,
    calib_start: int,
    calib_end:   int,
    save_tf:     bool,
) -> bool:
    """
    Run QDM for one model × scenario combination.

    File naming:
      Input  (historical) : pr_day_<model>_historical_<ensemble>_jakarta.nc
      Input  (future)     : pr_day_<model>_<scenario>_<ensemble>_jakarta.nc
      Output (corrected)  : pr_day_<model>_<scenario>_<ensemble>_jakarta_qdm.nc
    """
    cfg      = MODELS[model]
    ensemble = cfg["ensemble"]

    chirps_file = next(iter(sorted(output_path.glob("chirps_v2_jakarta_*.nc"))), None)
    hist_file   = output_path / f"pr_day_{model}_historical_{ensemble}_jakarta.nc"
    future_file = output_path / f"pr_day_{model}_{scenario}_{ensemble}_jakarta.nc"
    out_file    = output_path / f"pr_day_{model}_{scenario}_{ensemble}_jakarta_qdm.nc"

    # Check inputs exist
    missing = []
    if chirps_file is None:  missing.append("CHIRPS file (run prepare first)")
    if not hist_file.exists():   missing.append(str(hist_file.name))
    if not future_file.exists(): missing.append(str(future_file.name))
    if missing:
        for m in missing:
            logger.error(f"  Missing: {m}")
        return False

    logger.info("=" * 55)
    logger.info(f"QDM  |  model={model}  scenario={scenario}")
    logger.info(f"  Calibration : {calib_start}-{calib_end}")
    logger.info(f"  CHIRPS      : {chirps_file.name}")
    logger.info(f"  Historical  : {hist_file.name}")
    logger.info(f"  Future      : {future_file.name}")
    logger.info("=" * 55)

    # Load CHIRPS
    logger.info("Loading CHIRPS obs ...")
    ds_obs = xr.open_dataset(
        chirps_file,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    if "precip" in ds_obs and "pr" not in ds_obs:
        ds_obs = ds_obs.rename({"precip": "pr"})

    # Load historical model
    logger.info("Loading historical model ...")
    ds_hist_full = xr.open_dataset(
        hist_file,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    # Slice both to calibration period
    logger.info(f"Slicing to calibration period {calib_start}-{calib_end} ...")
    obs_years  = np.array([t.year for t in ds_obs.time.values])
    hist_years = np.array([t.year for t in ds_hist_full.time.values])

    obs_mask  = (obs_years  >= calib_start) & (obs_years  <= calib_end)
    hist_mask = (hist_years >= calib_start) & (hist_years <= calib_end)

    ds_obs_cal  = ds_obs.isel( time=np.where(obs_mask) [0])
    ds_hist_cal = ds_hist_full.isel(time=np.where(hist_mask)[0])

    logger.info(f"  CHIRPS cal days : {len(ds_obs_cal.time):,}")
    logger.info(f"  Hist   cal days : {len(ds_hist_cal.time):,}")

    if len(ds_obs_cal.time) == 0 or len(ds_hist_cal.time) == 0:
        logger.error("  Calibration period overlap is empty — check calib-start/end.")
        return False

    # Load future
    logger.info(f"Loading future ({scenario}) ...")
    ds_future = xr.open_dataset(
        future_file,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )
    logger.info(f"  Future days: {len(ds_future.time):,}")

    # Apply QDM
    logger.info("Applying QDM ...")
    da_corrected, transfers = apply_qdm_spatial(
        obs_ds    = ds_obs_cal,
        hist_ds   = ds_hist_cal,
        future_ds = ds_future,
        pr_var    = "pr",
    )

    # Save output
    ds_out = da_corrected.to_dataset(name="pr")
    ds_out.attrs.update({
        "title":           f"QDM bias-corrected {model} {scenario} precipitation",
        "model":           model,
        "ensemble":        ensemble,
        "scenario":        scenario,
        "bias_correction": "Quantile Delta Mapping (QDM) — Cannon et al. 2015",
        "reference_obs":   "CHIRPS v2.0",
        "calib_period":    f"{calib_start}-{calib_end}",
    })
    logger.info("Saving corrected output ...")
    save_model_streaming(ds_out, out_file, f"{model}/{scenario} QDM")

    # Optionally save transfer functions
    if save_tf:
        tf_path = output_path / f"transfer_functions_{model}_{scenario}_qdm.pkl"
        with open(tf_path, "wb") as fp:
            pickle.dump(transfers, fp)
        logger.info(f"  Transfer functions -> {tf_path.name}")

    # Make the summary
    raw_vals  = ds_future["pr"].values
    corr_vals = da_corrected.values
    raw_mean  = float(raw_vals[np.isfinite(raw_vals)].mean())
    corr_mean = float(corr_vals[np.isfinite(corr_vals)].mean())
    nan_count = int(np.isnan(corr_vals).sum())

    logger.info("=" * 55)
    logger.info(f"DONE: {out_file.name}")
    logger.info(f"  Raw future mean  : {raw_mean:.2f} mm/day")
    logger.info(f"  Corrected mean   : {corr_mean:.2f} mm/day")
    logger.info(f"  NaN cells        : {nan_count}")
    logger.info("=" * 55)

    ds_obs.close()
    ds_hist_full.close()
    ds_future.close()
    return True


# ===== CLI 'apply' subcommand ====================

@cli.command("apply")
@click.option("--model",    default="all",
              type=click.Choice(list(MODELS.keys()) + ["all"]), show_default=True,
              help="Which model to bias-correct, or 'all'.")
@click.option("--scenario", default="all",
              type=click.Choice(ALL_SCENARIOS + ["all"]), show_default=True,
              help="Which SSP scenario to correct, or 'all'.")
@click.option("--output-dir", default=str(DEFAULT_OUTPUT_DIR), show_default=True,
              help="Directory containing prepared files and where output is written.")
@click.option("--calib-start", default=CALIB_START, show_default=True,
              help="First year of calibration period.")
@click.option("--calib-end",   default=CALIB_END,   show_default=True,
              help="Last year of calibration period.")
@click.option("--save-transfer", is_flag=True, default=False,
              help="Save fitted transfer functions as .pkl files.")
def apply(model, scenario, output_dir, calib_start, calib_end, save_transfer):
    output_path  = Path(output_dir)
    models_to_run    = list(MODELS.keys()) if model    == "all" else [model]
    scenarios_to_run = ALL_SCENARIOS       if scenario == "all" else [scenario]

    completed = []
    failed    = []

    for mdl in models_to_run:
        for scenario in scenarios_to_run:
            if scenario not in MODELS[mdl]["scenarios"]:
                logger.info(f"Skipping {mdl}/{scenario} — not in model scenario list")
                continue

            logger.info(f"{'#' * 55}")
            logger.info(f"# Model: {mdl}  |  Scenario: {scenario}")
            logger.info(f"{'#' * 55}")

            ok = _run_qdm(
                model       = mdl,
                scenario    = scenario,
                output_path = output_path,
                calib_start = calib_start,
                calib_end   = calib_end,
                save_tf     = save_transfer,
            )
            if ok:
                completed.append((mdl, scenario))
            else:
                failed.append((mdl, scenario))

    logger.info("=" * 55)
    logger.info(f"DONE — {len(completed)} completed, {len(failed)} failed")
    logger.info("=" * 55)
    for mdl, scenario in completed:
        out = Path(output_dir) / f"pr_day_{mdl}_{scenario}_{MODELS[mdl]['ensemble']}_jakarta_qdm.nc"
        logger.info(f"  OK  {mdl:<15} {scenario:<10} -> {out.name}")
    if failed:
        logger.warning("Failed:")
        for mdl, scenario in failed:
            logger.warning(f"  !! {mdl:<15} {scenario}")


if __name__ == "__main__":
    cli()