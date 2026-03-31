import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional
import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===== Paths ====================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT    = PROJECT_ROOT.parent

DEFAULT_INDICES_DIR      = PROJECT_ROOT / "py" / "results" / "indices"
DEFAULT_WATER_STRESS_DIR = PROJECT_ROOT / "py" / "results" / "water_stress"
DEFAULT_WEIGHTS_PATH     = PROJECT_ROOT / "py" / "results" / "ensemble_weights.json"
DEFAULT_OUTPUT_DIR       = PROJECT_ROOT / "py" / "results" / "models"
DEFAULT_EROSIVITY_DIR    = PROJECT_ROOT / "py" / "results" / "erosivity"
DEFAULT_FIG_DIR          = PROJECT_ROOT / "py" / "results" / "figures"

# ===== CMIP6 registry ====================

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

# ===== Feature set ====================

# Core ETCCDI features used as RF predictors for R-proxy training.
FEATURES_CORE = ["PRCPTOT", "Rx1day", "Rx3day", "Rx5day", "WDF", "SDII"]

# Optional water-stress feature included if water_stress files are present
# Annual mean SPI-12 captures the multi-month moisture deficit signal
FEATURE_SPI   = "SPI12_mean"

# Full feature list (SPI12_mean appended if available; see load_indices_for_model)
FEATURES = FEATURES_CORE + [FEATURE_SPI]

# ===== R-factor proxy based on Bols (1978) Indonesia calibration ====================

R_ALPHA = 6.19
R_BETA  = 0.76

def compute_r_proxy(prcptot: np.ndarray, rx1day: np.ndarray) -> np.ndarray:
    """
    Bols (1978) R-factor proxy:
        R = 6.19 × PRCPTOT^(1-0.76) × Rx1day^0.76

    Used as the training target where the RF learns to reproduce this from the full feature set, capturing nonlinear inter-index interactions.
    """
    prcptot = np.asarray(prcptot, dtype=float)
    rx1day  = np.asarray(rx1day,  dtype=float)
    p_safe  = np.where(prcptot > 0, prcptot, np.nan)
    r_safe  = np.where(rx1day  > 0, rx1day,  np.nan)
    return R_ALPHA * (p_safe ** (1.0 - R_BETA)) * (r_safe ** R_BETA)

# ===== RF hyperparameters ====================

RF_PARAMS = dict(
    n_estimators      = 300,
    max_depth         = 12,
    min_samples_split = 5,
    min_samples_leaf  = 3,
    max_features      = "sqrt",
    bootstrap         = True,
    oob_score         = True,
    random_state      = 42,
    n_jobs            = -1,
)

CV_FOLDS     = 5
RANDOM_STATE = 42

# ===== Plotting constants ====================

FIGURE_DPI = 150
FONT_BASE  = 10

plt.rcParams.update({
    "font.size":         FONT_BASE,
    "axes.titlesize":    FONT_BASE + 1,
    "axes.labelsize":    FONT_BASE,
    "xtick.labelsize":   FONT_BASE - 1,
    "ytick.labelsize":   FONT_BASE - 1,
    "legend.fontsize":   FONT_BASE - 1,
    "figure.dpi":        FIGURE_DPI,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

SCENARIO_COLORS = {
    "historical": "#676767",
    "ssp126": "#2364a5",
    "ssp245": "#37b133",
    "ssp585": "#b02c25",
}
SCENARIO_LABELS = {
    "historical": "Historical",
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5",
}


# ===== Safe year extraction ====================

def _get_years(da: xr.DataArray) -> np.ndarray:
    if "year" in da.dims:
        return da.year.values.astype(int)
    time_vals = da.time.values
    if hasattr(time_vals[0], "year"):
        return np.array([t.year for t in time_vals], dtype=int)
    return pd.DatetimeIndex(time_vals).year.values.astype(int)


# ===== Ensemble weights ====================

def load_ensemble_weights(weights_path: Path) -> dict:
    """
    Load pre-computed ensemble weights from ensemble_validation.py output.
    Returns the 'raw' weights (pre-QDM dynamical skill) as recommended.
    Falls back to equal weights if the file is missing.
    """
    if not weights_path.exists():
        logger.warning(
            f"ensemble_weights.json not found at {weights_path}. "
            "Using equal weights (1/3 per model)."
        )
        n = len(MODELS)
        return {m: round(1.0 / n, 6) for m in MODELS}

    with open(weights_path) as fp:
        data = json.load(fp)

    weights = data.get("raw", {})
    logger.info("  Ensemble weights (pre-QDM dynamical skill):")
    for m, w in weights.items():
        logger.info(f"    {m:<15}: {w:.4f}")
    return weights


# ===== Data loading ====================

def _spatial_mean(da: xr.DataArray) -> float:
    """Spatial mean over all lat/lon dims, returning a scalar float."""
    val = da.values.astype(float)
    finite = val[np.isfinite(val)]
    return float(finite.mean()) if len(finite) > 0 else np.nan


def load_indices_for_model(
    model:            str,
    scenario:         str,
    indices_dir:      Path,
    water_stress_dir: Optional[Path] = None,
    temp_stat:        str = "median",
) -> Optional[pd.DataFrame]:
    """
    Load annual ETCCDI indices for one model × scenario and return a tidy DataFrame with columns: year, PRCPTOT, Rx1day, ..., SDII, SPI12_mean, R_proxy, scenario, model.

    Input filename (from precipitation_indices.py):
        {model}_{scenario}_{ensemble}_indices_jakarta.nc

    SPI12_mean is loaded from water_stress output if available:
        {model}_{scenario}_{ensemble}_water_stress_{temp_stat}_jakarta.nc
    """
    cfg      = MODELS[model]
    ensemble = cfg["ensemble"]

    # ===== Load ETCCDI indices ====================
    fname = f"{model}_{scenario}_{ensemble}_indices_jakarta.nc"
    fpath = indices_dir / fname

    if not fpath.exists():
        logger.warning(f"  Indices file not found: {fname} — skipping {model}/{scenario}")
        return None

    logger.info(f"  Loading indices : {fname}")
    ds = xr.open_dataset(
        fpath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
    )

    for var in FEATURES_CORE:
        if var not in ds:
            logger.error(f"  Variable '{var}' missing in {fname}")
            ds.close()
            return None

    years = ds["year"].values.astype(int)
    rows  = []

    for i, yr in enumerate(years):
        row = {"year": int(yr), "scenario": scenario, "model": model}
        for var in FEATURES_CORE:
            da  = ds[var].isel(year=i)
            row[var] = _spatial_mean(da)
        rows.append(row)

    ds.close()
    df = pd.DataFrame(rows)

    # ===== Load SPI12_mean from water_stress output ====================
    df[FEATURE_SPI] = np.nan
    if water_stress_dir is not None and water_stress_dir.exists():
        ws_fname = f"{model}_{scenario}_{ensemble}_water_stress_{temp_stat}_jakarta.nc"
        ws_fpath = water_stress_dir / ws_fname

        if ws_fpath.exists():
            logger.info(f"  Loading SPI12   : {ws_fname}")
            ds_ws = xr.open_dataset(
                ws_fpath,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
            )

            if "SPI12" in ds_ws:
                spi = ds_ws["SPI12"]
                # SPI12 is at monthly resolution; compute annual mean per year
                spi_years = _get_years(spi)
                for i, yr in enumerate(years):
                    mask = spi_years == yr
                    if mask.sum() > 0:
                        spi_yr = spi.isel(time=np.where(mask)[0])
                        # Spatial mean then time mean
                        vals = spi_yr.values.astype(float)
                        finite = vals[np.isfinite(vals)]
                        df.loc[df["year"] == yr, FEATURE_SPI] = (
                            float(finite.mean()) if len(finite) > 0 else np.nan
                        )
            else:
                logger.warning(f"  'SPI12' not found in {ws_fname}")
            ds_ws.close()
        else:
            logger.warning(f"  Water stress file not found: {ws_fname}")
            logger.warning(f"  SPI12_mean will be excluded from this model's feature set.")

    # ===== Compute R-proxy target ====================
    df["R_proxy"] = compute_r_proxy(df["PRCPTOT"].values, df["Rx1day"].values)
    df = df.dropna(subset=["R_proxy"])

    # Determine effective feature set (exclude SPI12_mean if all NaN)
    spi_available = df[FEATURE_SPI].notna().any()
    if not spi_available:
        logger.warning(f"  SPI12_mean unavailable for {model}/{scenario} — using core features only.")

    logger.info(
        f"  {model} {scenario:<12}: {len(df)} years | "
        f"R_proxy mean={df['R_proxy'].mean():.1f}  "
        f"max={df['R_proxy'].max():.1f}  "
        f"[MJ mm ha⁻¹ h⁻¹ yr⁻¹]  "
        f"SPI12_mean={'OK' if spi_available else 'MISSING'}"
    )
    return df


def get_effective_features(df: pd.DataFrame) -> list:
    """
    Return the feature list to use for this DataFrame.
    Drops SPI12_mean if it contains only NaN (water_stress not available).
    """
    if df[FEATURE_SPI].notna().any():
        return FEATURES
    return FEATURES_CORE


# ===== Model training & evaluation ====================

def train_and_evaluate(
    df_hist:   pd.DataFrame,
    rf_params: dict,
    cv_folds:  int,
    model_name: str = "",
) -> tuple:
    """
    Train an RF on df_hist and evaluate via k-fold CV + OOB score.

    Returns a tuple of (rf_model, metrics, cv_preds, features) where:
    rf_model   : fitted RandomForestRegressor (trained on all historical data)
    metrics    : dict of CV/OOB scores and feature importances
    cv_preds   : np.ndarray of cross-validated predictions (same length as df_hist)
    features   : list of feature names actually used (may exclude SPI12_mean)
    """
    features = get_effective_features(df_hist)
    X = df_hist[features].values
    y = df_hist["R_proxy"].values

    logger.info(f"  Features used ({len(features)}): {features}")
    logger.info(f"  Training samples: {len(y)}")
    logger.info(f"  Running {cv_folds}-fold CV...")

    kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rf_cv    = RandomForestRegressor(**rf_params)
    cv_preds = cross_val_predict(rf_cv, X, y, cv=kf)

    cv_rmse = float(np.sqrt(mean_squared_error(y, cv_preds)))
    cv_mae  = float(mean_absolute_error(y, cv_preds))
    cv_r2   = float(r2_score(y, cv_preds))
    logger.info(f"  CV  RMSE={cv_rmse:.2f}  MAE={cv_mae:.2f}  R²={cv_r2:.4f}")

    # Final model on all historical data
    logger.info("  Fitting final model on full historical data...")
    rf_model = RandomForestRegressor(**rf_params)
    rf_model.fit(X, y)
    oob_r2 = float(rf_model.oob_score_) if rf_params.get("oob_score") else np.nan
    logger.info(f"  OOB R² = {oob_r2:.4f}")

    # Permutation importance
    logger.info("  Computing permutation importance...")
    perm = permutation_importance(
        rf_model, X, y,
        n_repeats    = 30,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
    )

    metrics = {
        "cv_rmse":              cv_rmse,
        "cv_mae":               cv_mae,
        "cv_r2":                cv_r2,
        "oob_r2":               oob_r2,
        "features":             features,
        "perm_importances":     perm.importances_mean,
        "perm_importances_std": perm.importances_std,
        "impurity_importances": rf_model.feature_importances_,
        "model_name":           model_name,
    }

    return rf_model, metrics, cv_preds, features


# ===== Prediction ====================

def predict_scenario(
    rf_model:    RandomForestRegressor,
    df:          pd.DataFrame,
    features:    list,
    model_name:  str,
    scenario:    str,
) -> pd.DataFrame:
    """
    Apply a trained RF to one scenario's indices DataFrame.
    Returns a DataFrame with year, R_proxy_target, R_proxy_predicted columns.
    """
    X    = df[features].values
    pred = rf_model.predict(X)

    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        records.append({
            "model":              model_name,
            "scenario":           scenario,
            "year":               int(row["year"]),
            **{f: row[f] for f in features},
            "R_proxy_target":     row["R_proxy"],
            "R_proxy_predicted":  float(pred[i]),
        })
    return pd.DataFrame(records)


# ===== NetCDF output ====================

def save_predictions_nc(
    df_pred:      pd.DataFrame,
    model:        str,
    scenario:     str,
    out_dir:      Path,
) -> Path:
    """
    Save per-model R-factor predictions as a NetCDF file compatible with
    gloreda_scaling.py input expectations.

    Output variable: R_bols (raw Bols proxy from RF prediction)
    Dimension: year
    """
    cfg      = MODELS[model]
    ensemble = cfg["ensemble"]

    years  = df_pred["year"].values.astype(int)
    r_vals = df_pred["R_proxy_predicted"].values.astype(np.float32)

    da = xr.DataArray(
        r_vals,
        coords={"year": years},
        dims=["year"],
        attrs={
            "long_name":  "RF-predicted Bols R-factor proxy",
            "units":      "MJ mm ha-1 h-1 yr-1",
            "model":      model,
            "ensemble":   ensemble,
            "scenario":   scenario,
            "method":     "Random Forest regression on ETCCDI indices",
            "target":     "Bols (1978) R = 6.19 * PRCPTOT^0.24 * Rx1day^0.76",
        },
    )
    ds = da.to_dataset(name="R_bols")
    ds.attrs = {
        "title":    "RF-predicted Rainfall Erosivity R-factor — Jakarta",
        "model":    model,
        "ensemble": ensemble,
        "scenario": scenario,
    }

    out_name = f"R_bols_{model}_{scenario}_{ensemble}_jakarta.nc"
    out_path = out_dir / out_name
    ds.to_netcdf(
        out_path,
        encoding={"R_bols": {"dtype": "float32", "zlib": True, "complevel": 4}},
    )
    logger.info(f"  Saved NC: {out_name}")
    return out_path


# ===== Weighted ensemble aggregation ====================

def weighted_ensemble_mean(per_model: dict, weights: dict) -> np.ndarray:
    models = list(per_model.keys())
    arrays = np.stack([per_model[m] for m in models], axis=0)
    w      = np.array([weights.get(m, 1.0 / len(models)) for m in models])
    w      = w / w.sum()
    return np.einsum("m,m...->...", w, arrays)


def weighted_ensemble_std(per_model: dict, weights: dict) -> np.ndarray:
    mean   = weighted_ensemble_mean(per_model, weights)
    models = list(per_model.keys())
    arrays = np.stack([per_model[m] for m in models], axis=0)
    w      = np.array([weights.get(m, 1.0 / len(models)) for m in models])
    w      = w / w.sum()
    var    = np.einsum("m,m...->...", w, (arrays - mean) ** 2)
    return np.sqrt(var)


# ===== Figures ====================

def _add_metrics_box(ax, rmse, mae, r2, model_name=""):
    txt = f"RMSE = {rmse:.2f}\nMAE  = {mae:.2f}\nR²   = {r2:.4f}"
    if model_name:
        txt = f"{model_name}\n" + txt
    ax.text(
        0.05, 0.95, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=FONT_BASE - 1,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85),
        family="monospace",
    )


def plot_feature_importance(metrics: dict, fig_dir: Path, model: str) -> Path:
    features = metrics["features"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{model}  |  RF Feature Importance\nTarget: Bols (1978) R-factor proxy",
        fontsize=FONT_BASE + 1, fontweight="bold",
    )
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.85, len(features)))

    def _label_bars(ax, bars, vals, errs=None):
        x_max = ax.get_xlim()[1]
        for bar, v, err in zip(bars, vals, errs if errs is not None else [0] * len(vals)):
            x_anchor = v + (err if err else 0) + x_max * 0.02
            ax.text(
                x_anchor, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left",
                fontsize=FONT_BASE - 2, color="#333333",
            )
        ax.set_xlim(right=x_max * 1.18)

    # Permutation importance
    ax = axes[0]
    idx  = np.argsort(metrics["perm_importances"])
    vals = metrics["perm_importances"][idx]
    errs = metrics["perm_importances_std"][idx]
    feats = [features[i] for i in idx]
    bars = ax.barh(feats, vals, xerr=errs, color=colors[idx],
                   edgecolor="white", linewidth=0.5, capsize=3, height=0.55,
                   error_kw={"elinewidth": 1.2, "ecolor": "#555555"})
    ax.set_xlabel("Mean decrease in R² (permutation)")
    ax.set_title("Permutation Importance", fontweight="bold", pad=10)
    ax.axvline(0, color="#cccccc", lw=0.8, ls="--")
    ax.autoscale(axis="x")
    _label_bars(ax, bars, vals, errs)

    # MDI importance
    ax = axes[1]
    idx2  = np.argsort(metrics["impurity_importances"])
    vals2 = metrics["impurity_importances"][idx2]
    feats2 = [features[i] for i in idx2]
    bars2 = ax.barh(feats2, vals2, color=colors[idx2],
                    edgecolor="white", linewidth=0.5, height=0.55)
    ax.set_xlabel("Mean decrease in impurity (MDI)")
    ax.set_title("Impurity-Based Importance (MDI)", fontweight="bold", pad=10)
    ax.autoscale(axis="x")
    _label_bars(ax, bars2, vals2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / f"{model}_fig_feature_importance.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


def plot_observed_vs_predicted(
    df_hist:   pd.DataFrame,
    cv_preds:  np.ndarray,
    metrics:   dict,
    fig_dir:   Path,
    model:     str,
) -> Path:
    y_true = df_hist["R_proxy"].values
    years  = df_hist["year"].values

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(y_true, cv_preds, c=years, cmap="plasma",
                    s=55, edgecolors="white", linewidths=0.5, zorder=3)
    fig.colorbar(sc, ax=ax, pad=0.02).set_label("Year", fontsize=FONT_BASE - 1)

    lims = [min(y_true.min(), cv_preds.min()) * 0.95,
            max(y_true.max(), cv_preds.max()) * 1.05]
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1 line", zorder=2)
    ax.fill_between(lims, [v * 0.90 for v in lims], [v * 1.10 for v in lims],
                    alpha=0.12, color="#2166AC", label="±10% envelope")

    m, b = np.polyfit(y_true, cv_preds, 1)
    xs = np.linspace(lims[0], lims[1], 200)
    ax.plot(xs, m * xs + b, color="#b02c25", lw=1.4, label=f"Fit (slope={m:.2f})", zorder=4)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed R-factor proxy  [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_ylabel("Predicted R-factor proxy [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_title(
        f"{model}  |  Cross-Validated Observed vs. Predicted\n"
        f"Historical record  ({int(years.min())}–{int(years.max())})",
        fontweight="bold",
    )
    _add_metrics_box(ax, metrics["cv_rmse"], metrics["cv_mae"], metrics["cv_r2"])
    ax.legend(fontsize=FONT_BASE - 2)

    fig.tight_layout()
    out = fig_dir / f"{model}_fig_observed_vs_predicted.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


def plot_residuals(
    df_hist:  pd.DataFrame,
    cv_preds: np.ndarray,
    metrics:  dict,
    fig_dir:  Path,
    model:    str,
) -> Path:
    from scipy.stats import norm as scipy_norm

    y_true    = df_hist["R_proxy"].values
    years     = df_hist["year"].values
    residuals = cv_preds - y_true

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(cv_preds, residuals, c=years, cmap="plasma",
                s=45, edgecolors="white", linewidths=0.4, zorder=3)
    ax1.axhline(0, color="black", lw=1.1, ls="--")
    sort_idx = np.argsort(cv_preds)
    win = max(5, len(residuals) // 6)
    smoothed = pd.Series(residuals[sort_idx]).rolling(win, center=True, min_periods=3).mean()
    ax1.plot(cv_preds[sort_idx], smoothed, color="#b02c25", lw=1.4, label="Smooth trend")
    ax1.set_xlabel("Fitted values"); ax1.set_ylabel("Residual (pred − obs)")
    ax1.set_title("Residuals vs. Fitted", fontweight="bold")
    ax1.legend(fontsize=FONT_BASE - 2)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(years, residuals,
            color=np.where(residuals >= 0, "#2364a5", "#D73027"),
            width=0.8, edgecolor="white", linewidth=0.3)
    ax2.axhline(0, color="black", lw=1.0, ls="--")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Residual")
    ax2.set_title("Residuals over Time", fontweight="bold")
    ax2.legend(handles=[
        Line2D([0], [0], color="#2364a5", lw=0, marker="s", markersize=8, label="Over-predicted"),
        Line2D([0], [0], color="#b02c25", lw=0, marker="s", markersize=8, label="Under-predicted"),
    ], fontsize=FONT_BASE - 2)

    ax3 = fig.add_subplot(gs[2])
    mu, sigma = residuals.mean(), residuals.std()
    ax3.hist(residuals, bins=min(15, len(residuals) // 3),
             color="#37b133", edgecolor="white", linewidth=0.5,
             density=True, alpha=0.75, label="Residuals")
    xs = np.linspace(residuals.min() - sigma, residuals.max() + sigma, 300)
    ax3.plot(xs, scipy_norm.pdf(xs, mu, sigma),
             color="#676767", lw=1.8, ls="--", label=f"N({mu:.1f}, {sigma:.1f})")
    ax3.axvline(0, color="black", lw=1.0, ls=":")
    ax3.set_xlabel("Residual"); ax3.set_ylabel("Density")
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.legend(fontsize=FONT_BASE - 2)
    ax3.text(0.03, 0.72,
             f"Bias = {mu:.2f}\nσ    = {sigma:.2f}\nRMSE = {metrics['cv_rmse']:.2f}",
             transform=ax3.transAxes, va="top", ha="left",
             fontsize=FONT_BASE - 2, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85))

    fig.suptitle(f"{model}  |  Residual Diagnostics  ({CV_FOLDS}-fold CV, Historical)",
                 fontsize=FONT_BASE + 1, fontweight="bold")
    fig.tight_layout()
    out = fig_dir / f"{model}_fig_residuals.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


def plot_ensemble_projection(
    all_predictions: dict,
    weights:         dict,
    metrics_per_model: dict,
    fig_dir:         Path,
) -> Path:
    """
    Plot weighted ensemble R-factor projection with per-model lines and
    uncertainty band (weighted inter-model std).

    all_predictions : {model: {scenario: df_pred}}
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False,
                             gridspec_kw={"height_ratios": [2.2, 1]})

    ax = axes[0]

    # Per-model thin lines (background)
    for model in MODELS:
        if model not in all_predictions:
            continue
        for scen in ALL_SCENARIOS:
            if scen not in all_predictions[model]:
                continue
            df_s = all_predictions[model][scen].sort_values("year")
            color = SCENARIO_COLORS[scen]
            ax.plot(df_s["year"].values, df_s["R_proxy_predicted"].values,
                    color=color, lw=0.5, alpha=0.20)

    # Weighted ensemble mean + uncertainty band per scenario
    for scen in ALL_SCENARIOS:
        color = SCENARIO_COLORS[scen]
        label = SCENARIO_LABELS[scen]

        # Align all models on the same year axis
        year_sets = [
            set(all_predictions[m][scen]["year"].values.tolist())
            for m in MODELS
            if m in all_predictions and scen in all_predictions[m]
        ]
        if not year_sets:
            continue
        common_years = sorted(set.intersection(*year_sets))
        if not common_years:
            continue

        per_model_r = {}
        for model in MODELS:
            if model not in all_predictions or scen not in all_predictions[model]:
                continue
            df_s = all_predictions[model][scen].set_index("year")
            per_model_r[model] = np.array(
                [df_s.loc[yr, "R_proxy_predicted"] for yr in common_years],
                dtype=float,
            )

        if len(per_model_r) < 2:
            continue

        years_arr  = np.array(common_years)
        ens_mean   = weighted_ensemble_mean(per_model_r, weights)
        ens_std    = weighted_ensemble_std(per_model_r, weights)

        # 10-yr rolling smooth
        smooth_mean = pd.Series(ens_mean, index=years_arr).rolling(
            10, center=True, min_periods=5).mean().values
        smooth_std  = pd.Series(ens_std,  index=years_arr).rolling(
            10, center=True, min_periods=5).mean().values

        ax.plot(years_arr, smooth_mean, color=color, lw=2.3,
                label=label, zorder=4)
        ax.fill_between(years_arr,
                        smooth_mean - smooth_std,
                        smooth_mean + smooth_std,
                        color=color, alpha=0.15, zorder=3)

    ax.set_ylabel("R-factor proxy  [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_title(
        "Weighted Ensemble RF Projection — Jakarta Rainfall Erosivity\n"
        "10-yr rolling mean ± weighted inter-model σ  |  Bols (1978) R-factor",
        fontweight="bold",
    )
    ax.legend(loc="upper left", framealpha=0.9)

    # Skill summary box
    skill_lines = []
    for model, met in metrics_per_model.items():
        skill_lines.append(
            f"{model:<15}: CV R²={met['cv_r2']:.3f}  RMSE={met['cv_rmse']:.1f}"
        )
    ax.text(0.5, 0.02, "\n".join(skill_lines),
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FONT_BASE - 2, color="#696969", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    # Bottom panel: anomaly relative to ssp245 historical-period mean
    ax2 = axes[1]
    ref_vals = {}   # model -> mean R over HIST_PERIOD

    for model in MODELS:
        if model not in all_predictions:
            continue
        # Use ssp245 over HIST_PERIOD as reference (most data-rich overlap)
        for scen in ALL_SCENARIOS:
            if scen not in all_predictions[model]:
                continue
            df_ref = all_predictions[model][scen]
            hist_mask = (df_ref["year"] >= HIST_PERIOD[0]) & \
                        (df_ref["year"] <= HIST_PERIOD[1])
            if hist_mask.sum() > 0:
                ref_vals[model] = float(df_ref.loc[hist_mask, "R_proxy_predicted"].mean())
                break

    for scen in ALL_SCENARIOS:
        color = SCENARIO_COLORS[scen]
        year_sets = [
            set(all_predictions[m][scen]["year"].values.tolist())
            for m in MODELS
            if m in all_predictions and scen in all_predictions[m]
        ]
        if not year_sets:
            continue
        common_years = sorted(set.intersection(*year_sets))
        if not common_years:
            continue

        per_model_anom = {}
        for model in MODELS:
            if model not in all_predictions or scen not in all_predictions[model]:
                continue
            if model not in ref_vals:
                continue
            df_s = all_predictions[model][scen].set_index("year")
            r_arr = np.array([df_s.loc[yr, "R_proxy_predicted"] for yr in common_years])
            per_model_anom[model] = r_arr - ref_vals[model]

        if len(per_model_anom) < 2:
            continue

        years_arr = np.array(common_years)
        anom_mean = weighted_ensemble_mean(per_model_anom, weights)
        smoothed  = pd.Series(anom_mean, index=years_arr).rolling(
            10, center=True, min_periods=5).mean().values

        ax2.plot(years_arr, smoothed, color=color, lw=2.0,
                 label=SCENARIO_LABELS[scen])
        ax2.fill_between(years_arr, np.zeros_like(smoothed), smoothed,
                         color=color, alpha=0.12)

    ax2.axhline(0, color="#676767", lw=1.1, ls="--")
    ax2.set_ylabel("ΔR vs. historical mean")
    ax2.set_xlabel("Year")
    ax2.set_title("Change in Erosivity Relative to Historical Mean (10-yr smooth)",
                  fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=FONT_BASE - 2)

    fig.tight_layout()
    out = fig_dir / "fig_ensemble_projection.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ===== Metrics CSV ====================

def save_metrics(metrics: dict, out_dir: Path, model: str) -> Path:
    features = metrics["features"]
    rows = [
        {"metric": "CV_RMSE", "value": metrics["cv_rmse"], "notes": f"{CV_FOLDS}-fold CV"},
        {"metric": "CV_MAE",  "value": metrics["cv_mae"],  "notes": f"{CV_FOLDS}-fold CV"},
        {"metric": "CV_R2",   "value": metrics["cv_r2"],   "notes": f"{CV_FOLDS}-fold CV"},
        {"metric": "OOB_R2",  "value": metrics["oob_r2"],  "notes": "Out-of-bag (full model)"},
    ]
    for feat, imp, std in zip(features,
                               metrics["perm_importances"],
                               metrics["perm_importances_std"]):
        rows.append({"metric": f"PermImp_{feat}", "value": imp,
                     "notes": f"Permutation importance ± {std:.4f}"})
    for feat, imp in zip(features, metrics["impurity_importances"]):
        rows.append({"metric": f"MDI_{feat}", "value": imp,
                     "notes": "Mean decrease in impurity (MDI)"})

    df = pd.DataFrame(rows)
    out = out_dir / f"{model}_rf_metrics.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    logger.info(f"  Saved: {out.name}")
    return out


# ===== CLI ====================

@click.command()
@click.option("--indices-dir",
              default=str(DEFAULT_INDICES_DIR), show_default=True,
              help="Directory containing ETCCDI indices NetCDF files.")
@click.option("--water-stress-dir",
              default=str(DEFAULT_WATER_STRESS_DIR), show_default=True,
              help="Directory containing water_stress NetCDF files (for SPI12_mean).")
@click.option("--weights-path",
              default=str(DEFAULT_WEIGHTS_PATH), show_default=True,
              help="Path to ensemble_weights.json from ensemble_validation.py.")
@click.option("--output-dir",
              default=str(DEFAULT_OUTPUT_DIR), show_default=True,
              help="Directory to save RF model .pkl and metrics .csv files.")
@click.option("--erosivity-dir",
              default=str(DEFAULT_EROSIVITY_DIR), show_default=True,
              help="Directory to save per-model R-factor NetCDF outputs.")
@click.option("--fig-dir",
              default=str(DEFAULT_FIG_DIR), show_default=True,
              help="Directory to save figure .png files.")
@click.option("--model",
              default="all",
              type=click.Choice(list(MODELS.keys()) + ["all"]),
              show_default=True,
              help="Which CMIP6 model to train/predict, or 'all'.")
@click.option("--scenario",
              default="all",
              type=click.Choice(ALL_SCENARIOS + ["all"]),
              show_default=True,
              help="Which SSP scenario to predict, or 'all'.")
@click.option("--temp-stat",
              default="median",
              type=click.Choice(["median", "range_mid"]),
              show_default=True,
              help="IPCC AR6 warming stat used in water_stress filenames.")
@click.option("--n-estimators", default=RF_PARAMS["n_estimators"], show_default=True)
@click.option("--max-depth",    default=RF_PARAMS["max_depth"],    show_default=True)
@click.option("--cv-folds",     default=CV_FOLDS,                  show_default=True)
@click.option("--random-state", default=RANDOM_STATE,              show_default=True)
def main(
    indices_dir, water_stress_dir, weights_path, output_dir, erosivity_dir,
    fig_dir, model, scenario, temp_stat,
    n_estimators, max_depth, cv_folds, random_state,
):
    """
    Train per-model RFs and produce weighted ensemble erosivity projections.
    """
    indices_path      = Path(indices_dir)
    ws_path           = Path(water_stress_dir)
    out_path          = Path(output_dir)
    eros_path         = Path(erosivity_dir)
    fig_path          = Path(fig_dir)

    for p in [out_path, eros_path, fig_path]:
        p.mkdir(parents=True, exist_ok=True)

    if not indices_path.exists():
        logger.error(f"Indices directory not found: {indices_path}")
        logger.error("Run precipitation_indices.py --model all --scenario all first.")
        sys.exit(1)

    rf_params = {
        **RF_PARAMS,
        "n_estimators": n_estimators,
        "max_depth":    max_depth,
        "random_state": random_state,
    }

    models_to_run    = list(MODELS.keys()) if model    == "all" else [model]
    scenarios_to_run = ALL_SCENARIOS       if scenario == "all" else [scenario]

    # Load ensemble weights
    logger.info("=" * 60)
    logger.info("STEP 0: Loading ensemble weights")
    logger.info("=" * 60)
    weights = load_ensemble_weights(Path(weights_path))

    # Per-model training and prediction
    all_predictions  = {}   # {model: {scenario: df_pred}}
    metrics_per_model = {}  # {model: metrics}
    rf_models        = {}   # {model: rf}
    completed        = []
    skipped          = []

    for mdl in models_to_run:
        ensemble = MODELS[mdl]["ensemble"]
        logger.info(f"{'=' * 60}")
        logger.info(f"STEP 1–5: Model = {mdl}")
        logger.info(f"{'=' * 60}")

        # Load historical indices for training
        logger.info(f" --- [STEP 1] Loading historical indices for {mdl}...")

        # Historical data lives in the ssp files over HIST_PERIOD
        # Try each SSP scenario to find historical overlap
        df_hist = None
        for scen_try in scenarios_to_run:
            df_try = load_indices_for_model(
                model            = mdl,
                scenario         = scen_try,
                indices_dir      = indices_path,
                water_stress_dir = ws_path,
                temp_stat        = temp_stat,
            )
            if df_try is None:
                continue
            # Slice to historical period
            mask = (df_try["year"] >= HIST_PERIOD[0]) & \
                   (df_try["year"] <= HIST_PERIOD[1])
            df_slice = df_try[mask].copy()
            if len(df_slice) >= 10:
                df_hist = df_slice
                logger.info(
                    f"  Using {scen_try} for historical training slice: "
                    f"{len(df_hist)} years ({HIST_PERIOD[0]}–{HIST_PERIOD[1]})"
                )
                break

        if df_hist is None or len(df_hist) < 10:
            logger.error(
                f"  Insufficient historical data for {mdl} "
                f"(need >= 10 years in {HIST_PERIOD[0]}–{HIST_PERIOD[1]}). "
                "Skipping this model."
            )
            skipped.append(mdl)
            continue

        # Train RF on historical data
        logger.info(f" --- [STEP 2] Training RF for {mdl}...")
        rf_model, metrics, cv_preds, features = train_and_evaluate(
            df_hist, rf_params, cv_folds, model_name=mdl
        )
        rf_models[mdl]         = rf_model
        metrics_per_model[mdl] = metrics

        logger.info(
            f" --- {mdl} — OOB R²={metrics['oob_r2']:.4f}  "
            f"CV R²={metrics['cv_r2']:.4f}  "
            f"CV RMSE={metrics['cv_rmse']:.2f}"
        )
        logger.info("  Feature importances (permutation, descending):")
        for feat, imp, std in sorted(
            zip(features, metrics["perm_importances"], metrics["perm_importances_std"]),
            key=lambda x: x[1], reverse=True,
        ):
            logger.info(f"    {feat:<12}: {imp:.4f} ± {std:.4f}")

        # Predict all scenarios
        logger.info(f" --- [STEP 3] Predicting scenarios for {mdl}...")
        all_predictions[mdl] = {}

        for scen in scenarios_to_run:
            if scen not in MODELS[mdl]["scenarios"]:
                logger.info(f"  Skipping {mdl}/{scen} — not in model scenario list.")
                continue

            df_scen = load_indices_for_model(
                model            = mdl,
                scenario         = scen,
                indices_dir      = indices_path,
                water_stress_dir = ws_path,
                temp_stat        = temp_stat,
            )
            if df_scen is None:
                logger.warning(f"  No data for {mdl}/{scen} — skipping prediction.")
                continue

            df_pred = predict_scenario(rf_model, df_scen, features, mdl, scen)
            all_predictions[mdl][scen] = df_pred

            logger.info(
                f"  {scen:<12}: n={len(df_pred)}  "
                f"mean={df_pred['R_proxy_predicted'].mean():.1f}  "
                f"max={df_pred['R_proxy_predicted'].max():.1f}"
            )

            # Save NetCDF per model × scenario
            save_predictions_nc(df_pred, mdl, scen, eros_path)
            completed.append((mdl, scen))

        # Save RF model and diagnostics
        logger.info(f" --- [STEP 4] Saving outputs for {mdl}...")

        pkl_path = out_path / f"random_forest_erosivity_{mdl}.pkl"
        with open(pkl_path, "wb") as fp:
            pickle.dump({
                "model":     rf_model,
                "features":  features,
                "rf_params": rf_params,
                "metrics":   {k: v for k, v in metrics.items()
                              if not isinstance(v, np.ndarray)},
                "model_name": mdl,
                "hist_period": HIST_PERIOD,
            }, fp)
        logger.info(f"  Saved model: {pkl_path.name}")

        save_metrics(metrics, out_path, mdl)

        # Per-model figures
        logger.info(f" --- [STEP 5] Figures for {mdl}...")
        plot_feature_importance(metrics, fig_path, mdl)
        plot_observed_vs_predicted(df_hist, cv_preds, metrics, fig_path, mdl)
        plot_residuals(df_hist, cv_preds, metrics, fig_path, mdl)

    # Ensemble predictions CSV
    logger.info("=" * 60)
    logger.info("STEP 6: Ensemble aggregation")
    logger.info("=" * 60)

    all_dfs = []
    for mdl, scen_dict in all_predictions.items():
        for scen, df_pred in scen_dict.items():
            all_dfs.append(df_pred)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        ens_csv = out_path / "ensemble_rf_predictions.csv"
        df_all.to_csv(ens_csv, index=False, float_format="%.4f")
        logger.info(f"  Saved: {ens_csv.name}")

    # Ensemble projection figure
    if len(all_predictions) >= 2:
        plot_ensemble_projection(all_predictions, weights, metrics_per_model, fig_path)
    else:
        logger.warning("  Need >= 2 models for ensemble figure — skipping.")

    # Final summary
    logger.info("=" * 60)
    logger.info(f"DONE — {len(completed)} model × scenario predictions")
    logger.info("=" * 60)
    for mdl, scen in completed:
        ensemble = MODELS[mdl]["ensemble"]
        logger.info(
            f"  OK  {mdl:<15} {scen:<12} "
            f"w={weights.get(mdl, 1/3):.4f} "
            f"CV_R²={metrics_per_model.get(mdl, {}).get('cv_r2', float('nan')):.4f}"
        )
    if skipped:
        logger.info("Skipped models (insufficient historical data):")
        for mdl in skipped:
            logger.info(f"  --  {mdl}")

    logger.info(f"\nOutputs:")
    logger.info(f"  RF models   : {out_path}")
    logger.info(f"  Erosivity NC: {eros_path}")
    logger.info(f"  Figures     : {fig_path}")


if __name__ == "__main__":
    main()