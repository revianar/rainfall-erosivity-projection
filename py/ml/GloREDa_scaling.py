import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional
import click
import rasterio
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===== Paths ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT.parent                      # Jupyter Notebook/

DEFAULT_INPUT_DIR  = PROJECT_ROOT / "py" / "results" / "erosivity"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "py" / "results" / "erosivity_scaled"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "py" / "results"

# ===== CMIP6 registry =========================================================

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

# Historical period — must match what erosivity_rf.py used for its baseline
HIST_PERIOD = (1950, 2014)

# ===== GloREDa placeholder ====================================================

def extract_gloreda_mean(
    tif_path: Path,
    lat_min: float = -8.75,
    lat_max: float = -3.75,
    lon_min: float = 103.125,
    lon_max: float = 111.875,
) -> float:
    """
    Extract mean R-factor from GloREDa GeoTIFF over the Jakarta bounding box.

    Uses rasterio windowed reading — only the pixels inside the bounding box
    are loaded into memory. The full global raster (~3.7 GB) is never read.
    """
    from rasterio.windows import from_bounds

    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(
            f"GloREDa TIF not found: {tif_path}\n"
            "Download from: "
            "https://esdac.jrc.ec.europa.eu/content/global-rainfall-erosivity"
        )

    with rasterio.open(tif_path) as src:
        logger.info(f"  GloREDa TIF  : {tif_path.name}")
        logger.info(f"  CRS          : {src.crs}")
        logger.info(f"  Resolution   : {src.res}")
        logger.info(f"  Full size    : {src.width} x {src.height} px (NOT fully loaded)")

        # Read only the Jakarta bbox window — avoids loading the full global raster
        window = from_bounds(
            left=lon_min, bottom=lat_min, right=lon_max, top=lat_max,
            transform=src.transform,
        )
        data          = src.read(1, window=window)
        win_transform = src.window_transform(window)
        nodata        = src.nodata if src.nodata is not None else -9999.0

    # Build per-pixel coordinate grids from the window transform
    nrows, ncols     = data.shape
    col_idx, row_idx = np.meshgrid(np.arange(ncols), np.arange(nrows))
    lons, lats       = rasterio.transform.xy(
        win_transform,
        row_idx.ravel(),
        col_idx.ravel(),
    )
    lons = np.array(lons).reshape(nrows, ncols)
    lats = np.array(lats).reshape(nrows, ncols)

    valid = (
        (lats >= lat_min) & (lats <= lat_max) &
        (lons >= lon_min) & (lons <= lon_max) &
        np.isfinite(data) &
        (data > 0) &
        (data != nodata)
    )

    values = data[valid].astype(float)

    if len(values) == 0:
        raise ValueError(
            "No valid GloREDa pixels found in the Jakarta bounding box.\n"
            f"  Bbox: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]\n"
            f"  Window shape: {nrows} x {ncols} px\n"
            "Check TIF coverage and nodata value."
        )

    mean_val = float(values.mean())
    logger.info(f"  Window shape : {nrows} x {ncols} px loaded")
    logger.info(f"  Valid pixels : {len(values)}")
    logger.info(f"  R range      : {values.min():.1f} – {values.max():.1f}")
    logger.info(f"  GloREDa mean : {mean_val:.2f} MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    return mean_val


# ===== Safe year extraction ===================================================

def _get_years(da: xr.DataArray) -> np.ndarray:
    """Extract integer years — works with cftime and numpy datetime64."""
    time_vals = da.time.values if "time" in da.dims else da.year.values
    if hasattr(time_vals[0], "year"):
        return np.array([t.year for t in time_vals])
    return np.array(time_vals, dtype=int)


# ===== Core scaling logic =====================================================

def compute_scale_factor(
    da_r:        xr.DataArray,
    gloreda_r:   float,
    hist_start:  int = HIST_PERIOD[0],
    hist_end:    int = HIST_PERIOD[1],
    var_label:   str = "R_bols",
) -> dict:
    """
    Compute the multiplicative GloREDa scaling factor for one model.

    scale_factor = R_GloREDa / mean(R_Bols over historical period)

    Parameters
    ----------
    da_r       : DataArray of annual R-factor values with a 'year' dimension
    gloreda_r  : GloREDa R-factor for Jakarta [MJ·mm·ha⁻¹·h⁻¹·yr⁻¹]
    hist_start : first year of historical baseline
    hist_end   : last year of historical baseline
    var_label  : variable name for logging

    Returns
    -------
    dict with keys: bols_hist_mean, gloreda_r, scale_factor
    """
    years = _get_years(da_r)

    hist_mask = (years >= hist_start) & (years <= hist_end)
    if hist_mask.sum() == 0:
        raise ValueError(
            f"No years in historical period {hist_start}–{hist_end} "
            f"found in {var_label}. Available years: {years[0]}–{years[-1]}"
        )

    if "year" in da_r.dims:
        hist_vals = da_r.sel(year=slice(hist_start, hist_end)).values.astype(float)
    else:
        hist_vals = da_r.isel(time=np.where(hist_mask)[0]).values.astype(float)

    finite      = hist_vals[np.isfinite(hist_vals)]
    bols_mean   = float(finite.mean()) if len(finite) > 0 else np.nan

    if not np.isfinite(bols_mean) or bols_mean <= 0:
        raise ValueError(
            f"Historical mean of {var_label} is {bols_mean:.2f} — "
            "cannot compute scale factor. Check input file."
        )

    scale_factor = gloreda_r / bols_mean

    logger.info(f"    Bols historical mean ({hist_start}–{hist_end}): {bols_mean:.2f}")
    logger.info(f"    GloREDa R (Jakarta)                           : {gloreda_r:.2f}")
    logger.info(f"    Scale factor                                  : {scale_factor:.4f}")

    return {
        "bols_hist_mean": round(bols_mean,   4),
        "gloreda_r":      round(gloreda_r,   4),
        "scale_factor":   round(scale_factor, 6),
        "hist_period":    f"{hist_start}-{hist_end}",
    }


def apply_scaling(
    da_r:         xr.DataArray,
    scale_factor: float,
) -> xr.DataArray:
    """
    Apply multiplicative scaling to a DataArray of R-factor values.

    R_scaled(t) = R_Bols(t) × scale_factor

    Preserves all coordinates, dims, and attributes. Adds scaling metadata.
    """
    da_scaled = da_r * scale_factor
    da_scaled.attrs.update({
        **da_r.attrs,
        "scaling_method":  "GloREDa multiplicative anchoring (Option B)",
        "scale_factor":    float(scale_factor),
        "reference":       "Panagos et al. (2017) GloREDa dataset",
        "units":           "MJ mm ha-1 h-1 yr-1",
        "long_name":       "GloREDa-scaled Annual Rainfall Erosivity R-factor",
        "note": (
            "Absolute magnitude anchored to GloREDa via scale_factor = "
            "R_GloREDa / R_Bols_historical_mean. Relative change signal "
            "is identical to the unscaled Bols output."
        ),
    })
    return da_scaled


# ===== Per-model-scenario processor ====================

def load_hist_mean(
    model:      str,
    input_path: Path,
    hist_start: int = HIST_PERIOD[0],
    hist_end:   int = HIST_PERIOD[1],
) -> Optional[float]:
    """
    Load the historical R_bols file for a model and return the mean R
    over the historical baseline period.

    This is computed once per model and shared across all SSP scenarios,
    so the scale factor is always anchored to the same historical baseline
    regardless of which scenario is being processed.
    """
    cfg      = MODELS[model]
    ensemble = cfg["ensemble"]

    hist_fname = f"R_bols_{model}_historical_{ensemble}_jakarta.nc"
    hist_fpath = input_path / hist_fname

    if not hist_fpath.exists():
        logger.error(
            f"  Historical file not found: {hist_fname}\n"
            "  Scale factor cannot be computed — skipping this model."
        )
        return None

    logger.info(f"  Loading historical baseline: {hist_fname}")
    ds_hist = xr.open_dataset(
        hist_fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    r_vars = [v for v in ds_hist.data_vars if "R_bols" in v or "R" in v]
    if not r_vars:
        logger.error(f"  No R-factor variable in {hist_fname}.")
        ds_hist.close()
        return None

    da_hist = ds_hist[r_vars[0]]
    years   = da_hist.year.values.astype(int) if "year" in da_hist.dims \
              else np.array([t for t in da_hist.time.values], dtype=int)

    mask = (years >= hist_start) & (years <= hist_end)
    if mask.sum() == 0:
        logger.error(
            f"  No years in {hist_start}–{hist_end} found in {hist_fname}. "
            f"Available: {years[0]}–{years[-1]}"
        )
        ds_hist.close()
        return None

    if "year" in da_hist.dims:
        hist_vals = da_hist.sel(year=slice(hist_start, hist_end)).values.astype(float)
    else:
        hist_vals = da_hist.isel(time=np.where(mask)[0]).values.astype(float)

    finite    = hist_vals[np.isfinite(hist_vals)]
    bols_mean = float(finite.mean()) if len(finite) > 0 else np.nan
    ds_hist.close()

    if not np.isfinite(bols_mean) or bols_mean <= 0:
        logger.error(f"  Historical mean is {bols_mean} — cannot compute scale factor.")
        return None

    logger.info(
        f"  Historical Bols mean ({hist_start}–{hist_end}): {bols_mean:.2f} "
        "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"
    )
    return bols_mean


def process_one(
    model:          str,
    scenario:       str,
    gloreda_r:      float,
    bols_hist_mean: float,
    input_path:     Path,
    raw_path:       Path,
    scaled_path:    Path,
    hist_start:     int = HIST_PERIOD[0],
    hist_end:       int = HIST_PERIOD[1],
) -> Optional[dict]:
    """
    Load one model × scenario R-factor file, apply the pre-computed scale
    factor, and write raw copy and scaled output.

    The scale factor is computed externally from the historical file (via
    load_hist_mean) and passed in — so SSP files (which have no years in
    the historical period) are never used to derive the baseline.

    Parameters
    ----------
    bols_hist_mean : mean R over HIST_PERIOD from the historical file,
                     pre-computed by load_hist_mean()
    """
    cfg      = MODELS[model]
    ensemble = cfg["ensemble"]

    fname = f"R_bols_{model}_{scenario}_{ensemble}_jakarta.nc"
    fpath = input_path / fname

    if not fpath.exists():
        logger.warning(f"  File not found: {fpath.name} — skipping.")
        return None

    logger.info(f"  Loading: {fname}")
    ds = xr.open_dataset(
        fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    r_vars = [v for v in ds.data_vars if "R_bols" in v or "R" in v]
    if not r_vars:
        logger.error(f"  No R-factor variable found in {fname}. Skipping.")
        ds.close()
        return None

    logger.info(f"  Variables found: {r_vars}")

    # Scale factor is the same for all scenarios of this model
    scale_factor = gloreda_r / bols_hist_mean
    logger.info(f"    Bols hist mean  : {bols_hist_mean:.2f} ({hist_start}–{hist_end})")
    logger.info(f"    GloREDa R       : {gloreda_r:.2f}")
    logger.info(f"    Scale factor    : {scale_factor:.4f}")

    summary_rows = []
    ds_scaled    = ds.copy(deep=True)

    for var in r_vars:
        logger.info(f"\n  Processing variable: {var}")
        da_r    = ds[var]
        da_sc   = apply_scaling(da_r, scale_factor)
        sc_name = var.replace("R_bols", "R_gloreda_scaled")
        ds_scaled[sc_name] = da_sc

        summary_rows.append({
            "model":          model,
            "ensemble":       ensemble,
            "scenario":       scenario,
            "variable":       var,
            "bols_hist_mean": round(bols_hist_mean, 4),
            "gloreda_r":      round(gloreda_r,      4),
            "scale_factor":   round(scale_factor,   6),
            "hist_period":    f"{hist_start}-{hist_end}",
        })

    if not summary_rows:
        ds.close()
        return None

    # Write raw copy
    raw_fname = f"R_bols_raw_{model}_{scenario}_{ensemble}_jakarta.nc"
    raw_out   = raw_path / raw_fname
    ds.to_netcdf(
        raw_out,
        encoding={v: {"dtype": "float32", "zlib": True, "complevel": 4}
                  for v in ds.data_vars},
    )
    logger.info(f"  Raw copy saved  -> {raw_fname}")

    # Write scaled output
    scaled_fname = f"R_gloreda_scaled_{model}_{scenario}_{ensemble}_jakarta.nc"
    scaled_out   = scaled_path / scaled_fname

    ds_scaled.attrs.update({
        **ds.attrs,
        "model":             model,
        "ensemble":          ensemble,
        "scenario":          scenario,
        "gloreda_r_jakarta": gloreda_r,
        "bols_hist_mean":    bols_hist_mean,
        "scale_factor":      scale_factor,
        "scaling_method":    "GloREDa multiplicative anchoring (Option B)",
        "reference_gloreda": "Panagos et al. (2017) doi:10.1038/s41598-017-04282-8",
        "reference_bols":    "Bols (1978) Indonesia R-factor calibration",
    })

    ds_scaled.to_netcdf(
        scaled_out,
        encoding={v: {"dtype": "float32", "zlib": True, "complevel": 4}
                  for v in ds_scaled.data_vars},
    )
    logger.info(f"  Scaled output   -> {scaled_fname}")

    ds.close()
    ds_scaled.close()
    return summary_rows


# ===== CLI ====================

@click.command()
@click.option(
    "--input-dir",
    default=str(DEFAULT_INPUT_DIR),
    show_default=True,
    help="Directory containing erosivity_rf.py output files (R_bols_*.nc).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Root output directory. Raw and scaled subdirs are created inside.",
)
@click.option(
    "--results-dir",
    default=str(DEFAULT_RESULTS_DIR),
    show_default=True,
    help="Directory to save scale_factors.json and scaling_summary.csv.",
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
    help="Which SSP scenario to process, or 'all'.",
)
@click.option(
    "--gloreda-tif",
    default=str(DATA_ROOT / "GloREDa" / "GlobalR_NoPol.tif"),
    show_default=True,
    help="Path to GlobalR_NoPol.tif from the GloREDa dataset.",
)
@click.option(
    "--gloreda-r",
    default=None,
    type=float,
    show_default=True,
    help=(
        "GloREDa R-factor for Jakarta [MJ mm ha-1 h-1 yr-1]. "
        "Overrides the GLOREDA_R_JAKARTA constant in the script. "
        "Typical range for Jakarta: 8000-12000."
    ),
)
@click.option(
    "--hist-start",
    default=HIST_PERIOD[0],
    show_default=True,
    help="First year of historical baseline period for scale factor computation.",
)
@click.option(
    "--hist-end",
    default=HIST_PERIOD[1],
    show_default=True,
    help="Last year of historical baseline period for scale factor computation.",
)
def main(
    input_dir:   str,
    output_dir:  str,
    results_dir: str,
    model:       str,
    scenario:    str,
    gloreda_tif: str,
    gloreda_r:   Optional[float],
    hist_start:  int,
    hist_end:    int,
):
    # ===== Resolve GloREDa value at runtime ===================================
    if gloreda_r is not None:
        effective_gloreda_r = gloreda_r
        logger.info(f"Using --gloreda-r override: {effective_gloreda_r:.2f}")
    else:
        logger.info("=" * 65)
        logger.info("Extracting GloREDa R-factor from TIF (windowed read)...")
        logger.info("=" * 65)
        try:
            effective_gloreda_r = extract_gloreda_mean(Path(gloreda_tif))
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            logger.error(
                "\nAlternatively bypass TIF extraction with:\n"
                "  python GloREDa_scaling.py --gloreda-r <value>\n"
                "Typical range for Jakarta: 8000–12000 MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"
            )
            sys.exit(1)

    # ===== Set up directories ====================
    input_path  = Path(input_dir)
    out_path    = Path(output_dir)
    res_path    = Path(results_dir)
    raw_path    = out_path / "raw"
    scaled_path = out_path / "scaled"

    for p in [raw_path, scaled_path, res_path]:
        p.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        logger.error("Run erosivity_rf.py first to generate R_bols_*.nc files.")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("gloreda_scaling.py")
    logger.info("=" * 65)
    logger.info(f"Project root   : {PROJECT_ROOT}")
    logger.info(f"Input dir      : {input_path}")
    logger.info(f"Output dir     : {out_path}")
    logger.info(f"GloREDa R      : {effective_gloreda_r:.2f} MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    logger.info(f"Hist baseline  : {hist_start}–{hist_end}")

    models_to_run    = list(MODELS.keys()) if model    == "all" else [model]
    scenarios_to_run = ALL_SCENARIOS       if scenario == "all" else [scenario]

    all_rows   = []
    completed  = []
    skipped    = []
    sf_record  = {}   # model -> scenario -> scale_factor info

    for mdl in models_to_run:
        sf_record[mdl] = {}

        # Load historical baseline mean once per model
        # Used as the denominator for scale_factor across all scenarios of this model
        logger.info(f"{'=' * 65}")
        logger.info(f"Model: {mdl}  — loading historical baseline")
        logger.info(f"{'=' * 65}")
        bols_hist_mean = load_hist_mean(
            model      = mdl,
            input_path = input_path,
            hist_start = hist_start,
            hist_end   = hist_end,
        )
        if bols_hist_mean is None:
            logger.error(f"  Cannot process {mdl} — historical baseline unavailable.")
            for scen in scenarios_to_run:
                skipped.append((mdl, scen))
            continue

        for scen in scenarios_to_run:
            if scen not in MODELS[mdl]["scenarios"]:
                logger.info(f"\nSkipping {mdl}/{scen} — not in model scenario list.")
                skipped.append((mdl, scen))
                continue

            logger.info(f"\n{'=' * 65}")
            logger.info(f"Model: {mdl}  |  Scenario: {scen}")
            logger.info(f"{'=' * 65}")

            result = process_one(
                model          = mdl,
                scenario       = scen,
                gloreda_r      = effective_gloreda_r,
                bols_hist_mean = bols_hist_mean,
                input_path     = input_path,
                raw_path       = raw_path,
                scaled_path    = scaled_path,
                hist_start     = hist_start,
                hist_end       = hist_end,
            )

            if result is None:
                skipped.append((mdl, scen))
                continue

            all_rows.extend(result)
            completed.append((mdl, scen))

            # Record scale factors per variable for the JSON
            sf_record[mdl][scen] = {
                row["variable"]: {
                    "bols_hist_mean": row["bols_hist_mean"],
                    "gloreda_r":      row["gloreda_r"],
                    "scale_factor":   row["scale_factor"],
                    "hist_period":    row["hist_period"],
                }
                for row in result
            }

    # ===== Save scale factors JSON ====================
    sf_output = {
        "gloreda_r_jakarta": effective_gloreda_r,
        "reference": "Panagos et al. (2017) doi:10.1038/s41598-017-04282-8",
        "method": (
            "Multiplicative scaling: R_scaled(t) = R_Bols(t) × scale_factor. "
            "scale_factor = R_GloREDa / mean(R_Bols, hist_period). "
            "Relative change signal is preserved exactly."
        ),
        "scale_factors": sf_record,
    }

    sf_json_path = res_path / "gloreda_scale_factors.json"
    with open(sf_json_path, "w") as fp:
        json.dump(sf_output, fp, indent=2)
    logger.info(f"\n  Scale factors saved -> {sf_json_path}")

    # ===== Save summary CSV ====================
    if all_rows:
        df = pd.DataFrame(all_rows)

        # Add a % underestimation column — how far Bols was from GloREDa
        df["bols_underestimation_pct"] = (
            (df["gloreda_r"] - df["bols_hist_mean"]) / df["gloreda_r"] * 100
        ).round(2)

        csv_path = res_path / "gloreda_scaling_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"  Summary CSV saved   -> {csv_path}")

    # ===== Final summary ====================
    logger.info("\n" + "=" * 65)
    logger.info(f"DONE — {len(completed)} processed, {len(skipped)} skipped")
    logger.info("=" * 65)

    for mdl, scen in completed:
        ensemble = MODELS[mdl]["ensemble"]
        logger.info(
            f"  OK  {mdl:<15} {scen:<12} "
            f"raw -> R_bols_raw_{mdl}_{scen}_{ensemble}_jakarta.nc  "
            f"scaled -> R_gloreda_scaled_{mdl}_{scen}_{ensemble}_jakarta.nc"
        )
    if skipped:
        logger.info("Skipped:")
        for item in skipped:
            logger.info(f"  --  {item[0]:<15} {item[1]}")

    logger.info(f"\nOutput directory : {out_path}")
    logger.info(f"  raw/     : unmodified copies of erosivity_rf.py output")
    logger.info(f"  scaled/  : GloREDa-anchored R-factor projections")


if __name__ == "__main__":
    main()