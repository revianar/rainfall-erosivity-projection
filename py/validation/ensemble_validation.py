import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import pearsonr

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===== Paths ==================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT     = PROJECT_ROOT.parent

DEFAULT_RAW_DIR      = PROJECT_ROOT / "py" / "data" / "processed"
DEFAULT_BC_DIR       = PROJECT_ROOT / "py" / "data" / "bias_corrected"
DEFAULT_RESULTS_DIR  = PROJECT_ROOT / "py" / "results"
DEFAULT_CHIRPS_DIR   = DATA_ROOT / "CHIRPS"

# ===== CMIP6 registry =========================================================

MODELS = {
    "MRI-ESM2-0": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gn",
        "scenarios": ["ssp126", "ssp245", "ssp585"],
    },
    "EC-Earth3": {
        "ensemble":  "r1i1p1f1",
        "grid":      "gr",
        "scenarios": ["ssp126", "ssp245", "ssp585"],
    },
    "CNRM-CM6-1": {
        "ensemble":  "r1i1p1f2",
        "grid":      "gr",
        "scenarios": ["ssp126", "ssp245", "ssp585"],
    },
}

CALIB_START = 1981
CALIB_END   = 2014

WET_DAY_THRESHOLD = 1.0   # mm/day

# ===== Metric weights — motivated by R-factor exponent sensitivity =============
#
# R = 6.19 * PRCPTOT^0.76 * (Rx1day / SDII)^0.1
#
# PRCPTOT carries the dominant exponent (0.76), so errors there propagate most strongly into R.
# The Rx1day/SDII ratio contributes a smaller but nonlinear modifier.
# WDF controls wet/dry partitioning which shapes all intensity metrics.
# Seasonal cycle correlation captures monsoon phase — critical for inter-annual variability to be physically meaningful.

METRIC_WEIGHTS = {
    "PRCPTOT":        0.35,
    "Rx1day":         0.20,
    "SDII":           0.15,
    "WDF":            0.15,
    "seasonal_corr":  0.15,
}

# Sanity check
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9, "Metric weights must sum to 1.0"


# ===== Safe year extraction ===================================================

def _get_years(da: xr.DataArray) -> np.ndarray:
    """Extract integer years — works with cftime and numpy datetime64."""
    time_vals = da.time.values
    if hasattr(time_vals[0], "year"):
        return np.array([t.year for t in time_vals])
    return pd.DatetimeIndex(time_vals).year.values


def _get_months(da: xr.DataArray) -> np.ndarray:
    """Extract integer months — works with cftime and numpy datetime64."""
    time_vals = da.time.values
    if hasattr(time_vals[0], "month"):
        return np.array([t.month for t in time_vals])
    return pd.DatetimeIndex(time_vals).month.values


# ===== ETCCDI index computation ===============================================
# These are the same implementations as precipitation_indices.py
# duplicated here intentionally so this module is fully self-contained
# and importable without depending on the CLI scripts.

def _resample_yearly(da: xr.DataArray, func: str) -> xr.DataArray:
    """Cftime-safe yearly resampling via explicit year-loop groupby."""
    years        = _get_years(da)
    unique_years = np.unique(years)
    slices = []
    for yr in unique_years:
        da_yr = da.isel(time=np.where(years == yr)[0])
        if func == "sum":
            slices.append(da_yr.sum(dim="time"))
        elif func == "max":
            slices.append(da_yr.max(dim="time"))
        else:
            raise ValueError(f"Unknown func: {func!r}")
    return xr.concat(slices, dim=pd.Index(unique_years, name="year"))


def _compute_indices(pr: xr.DataArray) -> dict:
    """
    Compute all ETCCDI indices needed for validation.
    Returns a dict of {index_name: xr.DataArray} with a 'year' dimension.
    """
    threshold = WET_DAY_THRESHOLD

    wet_pr   = pr.where(pr >= threshold, 0.0)
    wet_flag = (pr >= threshold).astype(float)
    rx1      = pr
    rx5      = pr.rolling(time=5, center=False, min_periods=5).sum()

    prcptot      = _resample_yearly(wet_pr,   "sum")
    rx1day       = _resample_yearly(rx1,      "max")
    rx5day       = _resample_yearly(rx5,      "max")
    wdf          = _resample_yearly(wet_flag, "sum")
    annual_total = _resample_yearly(wet_pr,   "sum")
    annual_wdf   = _resample_yearly(wet_flag, "sum")
    sdii         = annual_total / annual_wdf.where(annual_wdf > 0)

    return {
        "PRCPTOT": prcptot,
        "Rx1day":  rx1day,
        "Rx5day":  rx5day,
        "WDF":     wdf,
        "SDII":    sdii,
    }


def _compute_seasonal_climatology(pr: xr.DataArray) -> np.ndarray:
    """
    12-element array of mean monthly precipitation over the full record.
    Used to assess monsoon phase skill (seasonal cycle correlation).
    """
    months       = _get_months(pr)
    climatology  = np.full(12, np.nan)
    for m in range(1, 13):
        vals = pr.isel(time=np.where(months == m)[0]).values.astype(float)
        finite = vals[np.isfinite(vals)]
        climatology[m - 1] = float(finite.mean()) if len(finite) > 0 else np.nan
    return climatology


# ===== Centre-grid extraction =================================================

CENTER_LAT = -7.5
CENTER_LON = 106.875


def _extract_center_cell(ds: xr.Dataset, pr_var: str = "pr") -> xr.DataArray:
    """
    Extract the single grid cell nearest to Jakarta city centre.
    Validation is performed at the point scale to avoid spatial averaging obscuring model skill differences.
    """
    pr = ds[pr_var] if pr_var in ds else ds["precip"]

    # Ensure non-negative and handle units
    if pr.attrs.get("units", "") in ["kg m-2 s-1", "kg/m2/s"]:
        logger.info("    Converting units: kg m-2 s-1 -> mm/day")
        pr = pr * 86400.0
        pr.attrs["units"] = "mm/day"

    pr = pr.clip(min=0.0)

    # Select nearest grid cell
    pr_cell = pr.sel(lat=CENTER_LAT, lon=CENTER_LON, method="nearest")
    actual_lat = float(pr_cell.lat.values) if "lat" in pr_cell.coords else CENTER_LAT
    actual_lon = float(pr_cell.lon.values) if "lon" in pr_cell.coords else CENTER_LON
    logger.info(f"    Centre cell: lat={actual_lat:.3f}, lon={actual_lon:.3f}")

    return pr_cell


# ===== Data loading ===========================================================

def _load_and_slice(
    fpath:      Path,
    start_year: int = CALIB_START,
    end_year:   int = CALIB_END,
    label:      str = "",
) -> xr.Dataset | None:
    """
    Open a NetCDF file with cftime decoding and slice to [start_year, end_year].
    Returns None if the file does not exist.
    """
    if not fpath.exists():
        logger.warning(f"  File not found: {fpath.name}  [{label}]")
        return None

    logger.info(f"  Loading: {fpath.name}  [{label}]")
    ds = xr.open_dataset(
        fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    # Rename precip variable
    if "precip" in ds and "pr" not in ds:
        ds = ds.rename({"precip": "pr"})

    # Slice to calibration period using year-safe masking
    pr_var = "pr" if "pr" in ds else list(ds.data_vars)[0]
    years  = _get_years(ds[pr_var])
    mask   = (years >= start_year) & (years <= end_year)
    ds     = ds.isel(time=np.where(mask)[0])

    n = int(mask.sum())
    if n == 0:
        logger.warning(f"  No data in {start_year}–{end_year} for {fpath.name}")
        return None

    logger.info(f"    Calibration days: {n:,}  ({start_year}–{end_year})")
    return ds


# ===== Skill metric computation ===============================================

def _nmae_skill(model_vals: np.ndarray, obs_vals: np.ndarray) -> float:
    """
    Normalised Mean Absolute Error skill score:
        skill = 1 - |mean(model) - mean(obs)| / |mean(obs)|

    Clipped to [0, 1]. A perfect model scores 1.0; a model with 100% bias
    scores 0.0; a model worse than climatology can score below 0 but is
    clipped to 0.
    """
    obs_mean   = float(np.nanmean(obs_vals))
    model_mean = float(np.nanmean(model_vals))

    if abs(obs_mean) < 1e-9:
        # Observed mean is effectively zero — skill undefined, return 0
        return 0.0

    nmae  = abs(model_mean - obs_mean) / abs(obs_mean)
    skill = float(np.clip(1.0 - nmae, 0.0, 1.0))
    return skill


def _seasonal_corr_skill(
    model_climatology: np.ndarray,
    obs_climatology:   np.ndarray,
) -> float:
    """
    Pearson correlation between model and observed 12-month climatology.
    Captures whether the model reproduces the phase and amplitude of the
    annual precipitation cycle (monsoon onset/withdrawal timing).

    Returns r clipped to [0, 1] — negative correlations score 0 since a
    model with inverted seasonality is worse than useless.
    """
    valid = np.isfinite(model_climatology) & np.isfinite(obs_climatology)
    if valid.sum() < 3:
        return 0.0
    r, _ = pearsonr(model_climatology[valid], obs_climatology[valid])
    return float(np.clip(r, 0.0, 1.0))


def _compute_skill_scores(
    model_indices:       dict,
    obs_indices:         dict,
    model_climatology:   np.ndarray,
    obs_climatology:     np.ndarray,
) -> dict:
    """
    Compute per-metric skill scores for one model against CHIRPS.

    Parameters
    ----------
    model_indices     : output of _compute_indices() for the model
    obs_indices       : output of _compute_indices() for CHIRPS
    model_climatology : 12-element monthly climatology array for model
    obs_climatology   : 12-element monthly climatology array for CHIRPS

    Returns
    -------
    dict of {metric_name: skill_score}  and raw bias info for the CSV
    """
    scores = {}
    details = {}

    for idx_name in ["PRCPTOT", "Rx1day", "SDII", "WDF"]:
        if idx_name not in model_indices or idx_name not in obs_indices:
            logger.warning(f"    Index {idx_name} missing — scoring as 0")
            scores[idx_name]  = 0.0
            details[idx_name] = {"model_mean": np.nan, "obs_mean": np.nan, "nmae": np.nan}
            continue

        m_vals = model_indices[idx_name].values.astype(float).ravel()
        o_vals = obs_indices[idx_name].values.astype(float).ravel()

        # Align on shared years
        m_years = model_indices[idx_name].year.values
        o_years = obs_indices[idx_name].year.values
        shared  = np.intersect1d(m_years, o_years)

        if len(shared) == 0:
            logger.warning(f"    No overlapping years for {idx_name} — scoring as 0")
            scores[idx_name]  = 0.0
            details[idx_name] = {"model_mean": np.nan, "obs_mean": np.nan, "nmae": np.nan}
            continue

        m_vals = model_indices[idx_name].sel(year=shared).values.astype(float).ravel()
        o_vals = obs_indices[idx_name].sel(year=shared).values.astype(float).ravel()

        finite = np.isfinite(m_vals) & np.isfinite(o_vals)
        m_vals = m_vals[finite]
        o_vals = o_vals[finite]

        skill      = _nmae_skill(m_vals, o_vals)
        obs_mean   = float(np.nanmean(o_vals))
        model_mean = float(np.nanmean(m_vals))
        nmae       = abs(model_mean - obs_mean) / max(abs(obs_mean), 1e-9)

        scores[idx_name]  = skill
        details[idx_name] = {
            "model_mean": round(model_mean, 3),
            "obs_mean":   round(obs_mean,   3),
            "nmae":       round(nmae,        4),
            "skill":      round(skill,       4),
        }

    # Seasonal cycle correlation
    scores["seasonal_corr"] = _seasonal_corr_skill(model_climatology, obs_climatology)
    details["seasonal_corr"] = {
        "model_mean": np.nan,
        "obs_mean":   np.nan,
        "nmae":       np.nan,
        "skill":      round(scores["seasonal_corr"], 4),
    }

    return scores, details


def _composite_score(skill_scores: dict) -> float:
    """
    Weighted sum of per-metric skill scores.
    Weights are defined in METRIC_WEIGHTS and motivated by R-factor sensitivity.
    """
    return float(sum(
        METRIC_WEIGHTS[metric] * skill_scores.get(metric, 0.0)
        for metric in METRIC_WEIGHTS
    ))


# ===== Main validation routine ================================================

def compute_ensemble_weights(
    raw_dir:     Path = DEFAULT_RAW_DIR,
    bc_dir:      Path = DEFAULT_BC_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    calib_start: int  = CALIB_START,
    calib_end:   int  = CALIB_END,
    save:        bool = True,
) -> dict:
    """
    Full ensemble validation pipeline. Computes skill scores and weights for
    both raw and QDM-corrected historical model runs against CHIRPS.

    Parameters
    ----------
    raw_dir     : directory containing crop_domain.py outputs
    bc_dir      : directory containing QDM.py outputs + CHIRPS file
    results_dir : where to save ensemble_weights.json and metrics CSV
    calib_start : first year of validation period
    calib_end   : last year of validation period
    save        : if True, write JSON and CSV to results_dir

    Returns
    -------
    dict with keys 'raw' and 'bc', each mapping model name -> weight (float)
    Example:
        {
            "raw": {"MRI-ESM2-0": 0.41, "EC-Earth3": 0.33, "CNRM-CM6-1": 0.26},
            "bc":  {"MRI-ESM2-0": 0.38, "EC-Earth3": 0.35, "CNRM-CM6-1": 0.27},
        }
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load CHIRPS ========================================================
    logger.info("=" * 65)
    logger.info("Loading CHIRPS observations ...")
    logger.info("=" * 65)

    chirps_path = next(iter(sorted(Path(bc_dir).glob("chirps_v2_jakarta_*.nc"))), None)
    if chirps_path is None:
        raise FileNotFoundError(
            f"CHIRPS file not found in {bc_dir}.\n"
            "Expected: chirps_v2_jakarta_1981_2014.nc\n"
            "Run QDM.py prepare first to generate it."
        )

    ds_chirps = _load_and_slice(chirps_path, calib_start, calib_end, "CHIRPS")
    if ds_chirps is None:
        raise ValueError("CHIRPS dataset is empty after slicing to calibration period.")

    pr_chirps      = _extract_center_cell(ds_chirps)
    obs_indices    = _compute_indices(pr_chirps)
    obs_clim       = _compute_seasonal_climatology(pr_chirps)

    logger.info(
        f"  CHIRPS centre cell: mean={float(pr_chirps.mean()):.2f} mm/day  "
        f"wet-days/yr={float(obs_indices['WDF'].mean()):.0f}"
    )

    # ===== Process each model =================================================
    all_rows   = []   # for the metrics CSV
    raw_scores = {}   # composite S_m before QDM
    bc_scores  = {}   # composite S_m after QDM

    for model, cfg in MODELS.items():
        ensemble = cfg["ensemble"]
        logger.info(f"{'=' * 65}")
        logger.info(f"Model: {model}  ({ensemble})")
        logger.info(f"{'=' * 65}")

        for run_type in ("raw", "bc"):
            if run_type == "raw":
                fname = f"pr_day_{model}_historical_{ensemble}_jakarta.nc"
                fpath = Path(raw_dir) / fname
                label = "pre-QDM"
            else:
                fname = f"pr_day_{model}_historical_{ensemble}_jakarta_qdm.nc"
                fpath = Path(bc_dir) / fname
                label = "post-QDM"

            logger.info(f"\n  [{label.upper()}]")
            ds = _load_and_slice(fpath, calib_start, calib_end, label)

            if ds is None:
                logger.warning(f"  Skipping {model} [{label}] — file missing.")
                if run_type == "raw":
                    raw_scores[model] = 0.0
                else:
                    bc_scores[model]  = 0.0
                continue

            pr_model    = _extract_center_cell(ds)
            mdl_indices = _compute_indices(pr_model)
            mdl_clim    = _compute_seasonal_climatology(pr_model)

            skill_scores, details = _compute_skill_scores(
                model_indices     = mdl_indices,
                obs_indices       = obs_indices,
                model_climatology = mdl_clim,
                obs_climatology   = obs_clim,
            )

            composite = _composite_score(skill_scores)

            logger.info(f"  Skill scores [{label}]:")
            for metric, w in METRIC_WEIGHTS.items():
                s = skill_scores.get(metric, 0.0)
                logger.info(f"    {metric:<16} skill={s:.4f}  weight={w:.2f}  contribution={s*w:.4f}")
            logger.info(f"  Composite S_m = {composite:.4f}")

            if run_type == "raw":
                raw_scores[model] = composite
            else:
                bc_scores[model]  = composite

            # Build a flat row for the CSV
            for metric, meta in details.items():
                all_rows.append({
                    "model":        model,
                    "ensemble":     ensemble,
                    "run_type":     label,
                    "metric":       metric,
                    "metric_weight": METRIC_WEIGHTS.get(metric, np.nan),
                    "model_mean":   meta.get("model_mean", np.nan),
                    "obs_mean":     meta.get("obs_mean",   np.nan),
                    "nmae":         meta.get("nmae",       np.nan),
                    "skill":        meta.get("skill",      np.nan),
                    "contribution": round(
                        METRIC_WEIGHTS.get(metric, 0.0) * meta.get("skill", 0.0), 4
                    ),
                    "composite_S":  round(composite, 4),
                })

            ds.close()
        ds_chirps.close()

    # ===== Normalise to weights ===============================================
    logger.info("=" * 65)
    logger.info("Normalising composite scores to ensemble weights ...")
    logger.info("=" * 65)

    def _normalise(scores: dict) -> dict:
        total = sum(scores.values())
        if total < 1e-9:
            # All models failed — fall back to equal weights
            logger.warning("All composite scores are zero — using equal weights.")
            n = len(scores)
            return {m: round(1.0 / n, 6) for m in scores}
        return {m: round(s / total, 6) for m, s in scores.items()}

    weights_raw = _normalise(raw_scores)
    weights_bc  = _normalise(bc_scores)

    logger.info("\n  Pre-QDM weights (dynamical skill):")
    for m, w in weights_raw.items():
        logger.info(f"    {m:<15} S={raw_scores[m]:.4f}  w={w:.4f}")

    logger.info("\n  Post-QDM weights (bias-corrected skill):")
    for m, w in weights_bc.items():
        logger.info(f"    {m:<15} S={bc_scores[m]:.4f}  w={w:.4f}")

    # ===== Save outputs =======================================================
    output = {
        "raw": weights_raw,
        "bc":  weights_bc,
        "metadata": {
            "calib_period":    f"{calib_start}-{calib_end}",
            "metric_weights":  METRIC_WEIGHTS,
            "weight_floor":    None,
            "note": (
                "Weights derived from multi-metric NMAE skill scoring on historical runs vs. CHIRPS v2.0."
                "'raw' = pre-QDM dynamical skill. 'bc' = post-QDM residual skill."
                "Downstream scripts should use 'raw' weights per Knutti et al. (2017) guidance "
                "performance weighting should reflect model physics, not correction fidelity."
            ),
        },
    }

    if save:
        weights_path = results_dir / "ensemble_weights.json"
        with open(weights_path, "w") as fp:
            json.dump(output, fp, indent=2)
        logger.info(f"\n  Weights saved -> {weights_path}")

        metrics_path = results_dir / "ensemble_validation_metrics.csv"
        df = pd.DataFrame(all_rows)

        # Add normalised weights as a column for convenience
        df["weight_raw"] = df["model"].map(weights_raw)
        df["weight_bc"]  = df["model"].map(weights_bc)

        df.to_csv(metrics_path, index=False)
        logger.info(f"  Metrics table saved -> {metrics_path}")

    logger.info("=" * 65)
    logger.info("Ensemble validation complete.")
    logger.info("=" * 65)

    return output


# ===== Convenience accessors ==================================================

def load_weights(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    which:       str  = "raw",
) -> dict:
    """Load saved ensemble weights from JSON."""
    weights_path = Path(results_dir) / "ensemble_weights.json"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"ensemble_weights.json not found in {results_dir}. "
            "Run compute_ensemble_weights() first."
        )
    with open(weights_path) as fp:
        data = json.load(fp)
    if which not in data:
        raise ValueError(f"Key '{which}' not in weights file. Choose 'raw' or 'bc'.")
    return data[which]


def weighted_ensemble_mean(
    per_model_arrays: dict,
    weights:          dict,
) -> np.ndarray:
    """
    Compute a weighted ensemble mean across models.

    Parameters:
    per_model_arrays : dict mapping model name -> np.ndarray (same shape)
    weights          : dict mapping model name -> float weight

    Returns:
    np.ndarray of the weighted mean (same shape as input arrays)

    Example:
    >>> r_predictions = {"MRI-ESM2-0": r_mri, "EC-Earth3": r_ec, "CNRM-CM6-1": r_cnrm}
    >>> weights = load_weights()
    >>> r_ensemble = weighted_ensemble_mean(r_predictions, weights)
    """
    models = list(per_model_arrays.keys())
    arrays = np.stack([per_model_arrays[m] for m in models], axis=0)
    w      = np.array([weights[m] for m in models])
    # Normalise weights in case subset of models is passed
    w      = w / w.sum()
    return float(np.sum(w[:, None] * arrays, axis=0)) if arrays.ndim == 1 \
        else np.einsum("m,m...->...", w, arrays)


def weighted_ensemble_std(
    per_model_arrays: dict,
    weights:          dict,
) -> np.ndarray:
    """
    Compute the weighted inter-model standard deviation (uncertainty envelope).

    Parameters
    ----------
    per_model_arrays : dict mapping model name -> np.ndarray (same shape)
    weights          : dict mapping model name -> float weight

    Returns
    -------
    np.ndarray of the weighted std (same shape as input arrays)
    """
    mean   = weighted_ensemble_mean(per_model_arrays, weights)
    models = list(per_model_arrays.keys())
    w      = np.array([weights[m] for m in models])
    w      = w / w.sum()
    arrays = np.stack([per_model_arrays[m] for m in models], axis=0)
    var    = np.einsum("m,m...->...", w, (arrays - mean) ** 2)
    return np.sqrt(var)


# ===== Entry point ============================================================

if __name__ == "__main__":
    compute_ensemble_weights()
