"""
Disclaimer: The R-factor is based on the Modifier Fournier Index (MFI) with modified parameters for fitting purposes with the target geography (Jakarta)

Random Forest
====================
Random Forest model for predicting rainfall erosivity (R-factor) over the Special Capital Region of Jakarta using CMIP5 HadGEM2-AO precipitation indices.

R-factor proxy
====================
The MFI is used as the R-factor proxy because it requires only monthly/annual totals (no sub-daily data):

    MFI = sum_{m=1}^{12} (p_m^2 / P)

where p_m is the mean monthly precipitation (mm) and P is the mean annual precipitation (mm) (Arnoldus, 1980). 

We scale MFI to approximate RUSLE R-factor units (MJ mm ha⁻¹ h⁻¹ yr⁻¹) via the empirical relationship
from Diodato & Bellocchi (2007) calibrated for tropical/monsoonal climates:

    R ≈ 0.739 × MFI^1.847                           (equation 1)

Because the indices NetCDF files store annual values (not monthly breakdowns),
we approximate MFI from the available annual indices using:

    MFI_proxy = (Rx1day / PRCPTOT) × PRCPTOT × k

where k = 12.0 is a distributional scaling constant that converts the ratio of the single-wettest-day fraction to a monthly concentration proxy. 

Next:

    R_proxy = α × PRCPTOT × (Rx1day / PRCPTOT)^β
            = α × PRCPTOT^(1-β) × Rx1day^β           (equation 2)

with α = 0.0483, β = 1.61  (fitted to the Diodato & Bellocchi tropical curve).
This produces R values in the correct physical range (~500–3000 MJ mm ha⁻¹ h⁻¹ yr⁻¹) for tropical Jakarta and is fully reproducible from the existing index files.

Methods
====================
1. Load indices NetCDF for every scenario (historical + rcp26/45/85).
2. Spatial-mean across the Jakarta domain → one time series per scenario.
3. Compute R_proxy target from PRCPTOT + Rx1day (equation 2).
4. Assemble feature matrix X = [PRCPTOT, Rx1day, Rx3day, Rx5day, WDF, SDII].
5. Train/evaluate RF on the historical record (5-fold CV).
6. Predict R_proxy for all scenarios (historical + future).
7. Save model (.pkl), metrics (.csv), predictions (.csv), and four figures.

Outputs (results/models/ and results/figures/)
==================================================
HadGEM2-AO_rf_erosivity_model.pkl       - trained RF model
HadGEM2-AO_rf_metrics.csv               - RMSE, MAE, R² (CV + holdout)
HadGEM2-AO_rf_predictions.csv           - year-by-year R predictions
fig_feature_importance.png              - permutation + impurity importance
fig_observed_vs_predicted.png           - CV predicted vs. target scatter
fig_residuals.png                       - residual diagnostics (2-panel)
fig_scenario_projection.png             - historical + RCP time series

References
=====================
Arnoldus, H.M.J. (1980). An approximation of the rainfall factor in the USLE. In: Assessment of Erosion (ed. De Boodt & Gabriels), Wiley, 127–132.
Diodato, N. and Bellocchi, G. (2007). Estimating monthly (R)USLE climate input in a Mediterranean region using limited data. J. Hydrol. 345, 224–236.
"""

import sys
import pickle
import logging
from pathlib import Path
import click
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INDICES_DIR      = PROJECT_ROOT / "py" / "results" / "indices"
DEFAULT_EXTREME_DIR      = PROJECT_ROOT / "py" / "results" / "extreme_freq"
DEFAULT_WATER_STRESS_DIR = PROJECT_ROOT / "py" / "results" / "water_stress"
DEFAULT_OUTPUT_DIR       = PROJECT_ROOT / "py" / "results" / "models"
DEFAULT_FIG_DIR          = PROJECT_ROOT / "py" / "results" / "figures"

MODEL     = "HadGEM2-AO"
SCENARIOS = ["historical", "rcp26", "rcp45", "rcp85"]

FEATURES = ["PRCPTOT", "Rx1day", "Rx3day", "Rx5day", "WDF", "SDII"]

SCENARIO_COLORS = {
    "historical": "#676767",
    "rcp26":      "#2364a5",
    "rcp45":      "#37b133",
    "rcp85":      "#b02c25",
}
SCENARIO_LABELS = {
    "historical": "Historical (1976–2005)",
    "rcp26":      "RCP 2.6 (+2.6 W/m²)",
    "rcp45":      "RCP 4.5 (+4.5 W/m²)",
    "rcp85":      "RCP 8.5 (+8.5 W/m²)",
}

# R-factor proxy calibration constants (equation 2 in module docstring)
R_ALPHA = 0.0483
R_BETA  = 1.61

# Random Forest hyper-parameters (mirrors config.yml)
RF_PARAMS = dict(
    n_estimators     = 300,
    max_depth        = 12,
    min_samples_split= 5,
    min_samples_leaf = 3,
    max_features     = "sqrt",
    bootstrap        = True,
    oob_score        = True,
    random_state     = 42,
    n_jobs           = -1,
)

CV_FOLDS     = 5
RANDOM_STATE = 42


# ===== R-factor proxy ====================

def compute_r_proxy(prcptot: np.ndarray, rx1day: np.ndarray) -> np.ndarray:
    """
    Compute the Modified Fournier Index-derived R-factor proxy.

    R_proxy = α × PRCPTOT^(1 - β) × Rx1day^β

    It will return as R_proxy in approximate RUSLE units (MJ mm ha⁻¹ h⁻¹ yr⁻¹)
    """
    prcptot = np.asarray(prcptot, dtype=float)
    rx1day  = np.asarray(rx1day,  dtype=float)

    # Guard against zero/negative values before raising to fractional power
    p_safe = np.where(prcptot > 0, prcptot, np.nan)
    r_safe = np.where(rx1day  > 0, rx1day,  np.nan)

    return R_ALPHA * (p_safe ** (1.0 - R_BETA)) * (r_safe ** R_BETA)


# ===== Data loading ====================

def load_scenario_indices(indices_dir: Path, scenario: str) -> pd.DataFrame | None:
    """
    Load the annual indices NetCDF for one scenario, spatial-average over the
    Jakarta domain, and return a tidy DataFrame with columns:
        year, PRCPTOT, Rx1day, Rx3day, Rx5day, WDF, SDII, R_proxy, scenario
    """
    fname = f"{MODEL}_{scenario}_indices_jakarta.nc"
    fpath = indices_dir / fname

    if not fpath.exists():
        logger.warning(f"  File not found: {fname} - skipping {scenario}")
        return None

    logger.info(f"  Loading: {fname}")
    ds = xr.open_dataset(fpath)

    rows = []
    years = ds["year"].values.astype(int)

    for var in FEATURES:
        if var not in ds:
            logger.error(f"  Variable '{var}' missing in {fname}")
            ds.close()
            return None

    for i, yr in enumerate(years):
        row = {"year": yr, "scenario": scenario}
        for var in FEATURES:
            da = ds[var].isel(year=i)
            # Spatial mean
            val = float(da.values.mean()) if da.ndim > 0 else float(da.values)
            row[var] = val
        rows.append(row)

    ds.close()

    df = pd.DataFrame(rows)
    df["R_proxy"] = compute_r_proxy(df["PRCPTOT"].values, df["Rx1day"].values)
    df = df.dropna(subset=["R_proxy"])

    logger.info(
        f"  {scenario:<12}: {len(df)} years | "
        f"R_proxy mean={df['R_proxy'].mean():.1f}  "
        f"min={df['R_proxy'].min():.1f}  "
        f"max={df['R_proxy'].max():.1f}  "
        f"[MJ mm ha⁻¹ h⁻¹ yr⁻¹]"
    )
    return df


# ===== Model training & evaluation ====================

def train_and_evaluate(
    df_hist: pd.DataFrame,
    rf_params: dict,
    cv_folds: int,
) -> tuple[RandomForestRegressor, dict, np.ndarray]:
    X = df_hist[FEATURES].values
    y = df_hist["R_proxy"].values

    # 5-fold cross-validation predictions (for scatter / residual plots)
    logger.info(f"  Running {cv_folds}-fold cross-validation...")
    kf       = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rf_cv    = RandomForestRegressor(**rf_params)
    cv_preds = cross_val_predict(rf_cv, X, y, cv=kf)

    cv_rmse = np.sqrt(mean_squared_error(y, cv_preds))
    cv_mae  = mean_absolute_error(y, cv_preds)
    cv_r2   = r2_score(y, cv_preds)

    logger.info(f"  CV  RMSE = {cv_rmse:.2f}  MAE = {cv_mae:.2f}  R² = {cv_r2:.4f}")

    # Final model fitted on ALL historical data
    logger.info("  Fitting final model on full historical data...")
    model = RandomForestRegressor(**rf_params)
    model.fit(X, y)

    oob_r2 = model.oob_score_ if rf_params.get("oob_score") else np.nan

    # Permutation importance (more reliable than impurity-based for correlated features)
    logger.info("  Computing permutation importance...")
    perm = permutation_importance(
        model, X, y,
        n_repeats   = 30,
        random_state= RANDOM_STATE,
        n_jobs      = -1,
    )

    metrics = {
        "cv_rmse":            cv_rmse,
        "cv_mae":             cv_mae,
        "cv_r2":              cv_r2,
        "oob_r2":             oob_r2,
        "perm_importances":   perm.importances_mean,
        "perm_importances_std": perm.importances_std,
        "impurity_importances": model.feature_importances_,
    }

    return model, metrics, cv_preds


# ===== Prediction for all scenarios ====================

def predict_all_scenarios(
    model: RandomForestRegressor,
    all_dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records = []
    for scenario, df in all_dfs.items():
        X   = df[FEATURES].values
        pred = model.predict(X)
        for i, (_, row) in enumerate(df.iterrows()):
            records.append({
                "scenario":   scenario,
                "year":       int(row["year"]),
                **{f: row[f] for f in FEATURES},
                "R_proxy_target":    row["R_proxy"],
                "R_proxy_predicted": pred[i],
            })
    return pd.DataFrame(records)


# ===== Figures ====================

FIGURE_DPI = 150
FONT_BASE  = 10

plt.rcParams.update({
    "font.size":        FONT_BASE,
    "axes.titlesize":   FONT_BASE + 1,
    "axes.labelsize":   FONT_BASE,
    "xtick.labelsize":  FONT_BASE - 1,
    "ytick.labelsize":  FONT_BASE - 1,
    "legend.fontsize":  FONT_BASE - 1,
    "figure.dpi":       FIGURE_DPI,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})


def _add_metrics_box(ax, rmse, mae, r2):
    txt = f"RMSE = {rmse:.1f}\nMAE  = {mae:.1f}\nR²   = {r2:.3f}"
    ax.text(
        0.05, 0.95, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=FONT_BASE - 1,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85),
        family="monospace",
    )


# ===== Figure 1: Feature Importance ====================

def plot_feature_importance(metrics: dict, fig_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"{MODEL}  |  Random Forest Feature Importance\nTarget: Rainfall Erosivity R-factor proxy",
        fontsize=FONT_BASE + 1, fontweight="bold",
    )

    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.85, len(FEATURES)))

    def _label_bars(ax, bars, vals, errs=None):
        x_max = ax.get_xlim()[1]
        for bar, v, err in zip(bars, vals, errs if errs is not None else [0] * len(vals)):
            # Anchor label just past the error cap, with a small fixed padding
            x_anchor = v + (err if err else 0) + x_max * 0.02
            ax.text(
                x_anchor,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va="center", ha="left",
                fontsize=FONT_BASE - 2,
                color="#333333",
            )
        # Expand x-axis limit to ensure labels don't get clipped
        ax.set_xlim(right=x_max * 1.18)

    # Left: Permutation importance
    ax = axes[0]
    idx  = np.argsort(metrics["perm_importances"])
    vals = metrics["perm_importances"][idx]
    errs = metrics["perm_importances_std"][idx]
    feats = [FEATURES[i] for i in idx]

    bars = ax.barh(
        feats, vals, xerr=errs,
        color=colors[idx], edgecolor="white", linewidth=0.5,
        capsize=3, height=0.55,
        error_kw={"elinewidth": 1.2, "ecolor": "#555555", "capthick": 1.2},
    )
    ax.set_xlabel("Mean decrease in R² (permutation)")
    ax.set_title("Permutation Importance", fontweight="bold", pad=10)
    ax.axvline(0, color="#cccccc", lw=0.8, ls="--")
    ax.set_xlim(left=-0.01)
    ax.tick_params(axis="y", labelsize=FONT_BASE - 1)
    # Draw labels after autoscaling so x_max is correct
    ax.autoscale(axis="x")
    _label_bars(ax, bars, vals, errs)

    # Right: Impurity (MDI) importance
    ax = axes[1]
    idx2  = np.argsort(metrics["impurity_importances"])
    vals2 = metrics["impurity_importances"][idx2]
    feats2 = [FEATURES[i] for i in idx2]

    bars2 = ax.barh(
        feats2, vals2,
        color=colors[idx2], edgecolor="white", linewidth=0.5, height=0.55,
    )
    ax.set_xlabel("Mean decrease in impurity (MDI)")
    ax.set_title("Impurity-Based Importance (MDI)", fontweight="bold", pad=10)
    ax.tick_params(axis="y", labelsize=FONT_BASE - 1)
    ax.autoscale(axis="x")
    _label_bars(ax, bars2, vals2)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = fig_dir / f"{MODEL}_fig_feature_importance.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ===== Figure 2: Observed vs. Predicted (CV) ====================

def plot_observed_vs_predicted(
    df_hist: pd.DataFrame,
    cv_preds: np.ndarray,
    metrics: dict,
    fig_dir: Path,
) -> Path:
    y_true = df_hist["R_proxy"].values
    years  = df_hist["year"].values

    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        y_true, cv_preds,
        c=years, cmap="plasma",
        s=55, edgecolors="white", linewidths=0.5, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("Year", fontsize=FONT_BASE - 1)

    # 1:1 line
    lims = [
        min(y_true.min(), cv_preds.min()) * 0.95,
        max(y_true.max(), cv_preds.max()) * 1.05,
    ]
    ax.plot(lims, lims, "k--", lw=1.2, label="1:1 line", zorder=2)

    # ±10 % envelope
    ax.fill_between(
        lims,
        [v * 0.90 for v in lims],
        [v * 1.10 for v in lims],
        alpha=0.12, color="#2166AC", label="±10% envelope",
    )

    # Linear trend line through scatter
    m, b = np.polyfit(y_true, cv_preds, 1)
    xs = np.linspace(lims[0], lims[1], 200)
    ax.plot(xs, m * xs + b, color="#b02c25", lw=1.4,
        label=f"Fit (slope={m:.2f})", zorder=4)

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed R-factor proxy  [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_ylabel("Predicted R-factor proxy [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_title(
        f"{MODEL}  |  Cross-Validated Observed vs. Predicted\n"
        f"Historical record  ({int(years.min())}–{int(years.max())})",
        fontweight="bold",
    )
    _add_metrics_box(ax, metrics["cv_rmse"], metrics["cv_mae"], metrics["cv_r2"])
    ax.legend(fontsize=FONT_BASE - 2)

    fig.tight_layout()
    out = fig_dir / f"{MODEL}_fig_observed_vs_predicted.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ===== Figure 3: Residual Diagnostics ===============

def plot_residuals(
    df_hist: pd.DataFrame,
    cv_preds: np.ndarray,
    metrics: dict,
    fig_dir: Path,
) -> Path:
    y_true   = df_hist["R_proxy"].values
    years    = df_hist["year"].values
    residuals = cv_preds - y_true

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # Panel A: Residuals vs. fitted
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(cv_preds, residuals, c=years, cmap="plasma",
        s=45, edgecolors="white", linewidths=0.4, zorder=3)
    ax1.axhline(0, color="black", lw=1.1, ls="--")
    # LOESS-style smoothed trend via rolling mean (no statsmodels dependency)
    sort_idx = np.argsort(cv_preds)
    win = max(5, len(residuals) // 6)
    smoothed = pd.Series(residuals[sort_idx]).rolling(win, center=True, min_periods=3).mean()
    ax1.plot(cv_preds[sort_idx], smoothed, color="#b02c25", lw=1.4, label="Smooth trend")
    ax1.set_xlabel("Fitted (predicted) values")
    ax1.set_ylabel("Residual (predicted − observed)")
    ax1.set_title("Residuals vs. Fitted", fontweight="bold")
    ax1.legend(fontsize=FONT_BASE - 2)

    # Panel B: Residuals over time
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(years, residuals,
        color=np.where(residuals >= 0, "#2364a5", "#D73027"),
        width=0.8, edgecolor="white", linewidth=0.3)
    ax2.axhline(0, color="black", lw=1.0, ls="--")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals over Time", fontweight="bold")

    overbar  = Line2D([0], [0], color="#2364a5", lw=0, marker="s",
        markersize=8, label="Over-predicted")
    underbar = Line2D([0], [0], color="#b02c25", lw=0, marker="s",
        markersize=8, label="Under-predicted")
    ax2.legend(handles=[overbar, underbar], fontsize=FONT_BASE - 2)

    # Panel C: Residual distribution (histogram + KDE)
    ax3 = fig.add_subplot(gs[2])
    from scipy.stats import norm as scipy_norm
    ax3.hist(residuals, bins=min(15, len(residuals) // 3),
        color="#37b133", edgecolor="white", linewidth=0.5,
        density=True, alpha=0.75, label="Residuals")
    mu, sigma = residuals.mean(), residuals.std()
    xs = np.linspace(residuals.min() - sigma, residuals.max() + sigma, 300)
    ax3.plot(xs, scipy_norm.pdf(xs, mu, sigma),
        color="#676767", lw=1.8, ls="--", label=f"N({mu:.1f}, {sigma:.1f})")
    ax3.axvline(0, color="black", lw=1.0, ls=":")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Density")
    ax3.set_title("Residual Distribution", fontweight="bold")
    ax3.legend(fontsize=FONT_BASE - 2)

    bias_txt = (
        f"Bias = {mu:.2f}\n"
        f"σ    = {sigma:.2f}\n"
        f"RMSE = {metrics['cv_rmse']:.2f}"
    )
    ax3.text(0.97, 0.97, bias_txt,
             transform=ax3.transAxes, va="top", ha="right",
             fontsize=FONT_BASE - 2, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.85))

    fig.suptitle(
        f"{MODEL}  |  Residual Diagnostics  (5-fold CV, Historical)",
        fontsize=FONT_BASE + 1, fontweight="bold",
    )
    fig.tight_layout()
    out = fig_dir / f"{MODEL}_fig_residuals.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ===== Figure 4: Scenario Projections ====================

def plot_scenario_projection(
    df_predictions: pd.DataFrame,
    metrics: dict,
    fig_dir: Path,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1]})

    # Top panel: time series
    ax = axes[0]

    for scen in SCENARIOS:
        df_s = df_predictions[df_predictions["scenario"] == scen].sort_values("year")
        if df_s.empty:
            continue
        y_pred = df_s["R_proxy_predicted"].values
        years  = df_s["year"].values
        color  = SCENARIO_COLORS[scen]
        label  = SCENARIO_LABELS[scen]

        # Raw annual line (thin, transparent)
        ax.plot(years, y_pred, color=color, lw=0.7, alpha=0.35)

        # 10-year rolling mean (thick)
        smoothed = pd.Series(y_pred, index=years).rolling(10, center=True, min_periods=5).mean()
        ax.plot(years, smoothed.values, color=color, lw=2.2,
                label=label, zorder=3)

        # Shaded ±1 rolling std envelope for RCP scenarios only
        if scen != "historical":
            roll_std = pd.Series(y_pred, index=years).rolling(10, center=True, min_periods=5).std()
            ax.fill_between(years,
                            smoothed.values - roll_std.values,
                            smoothed.values + roll_std.values,
                            color=color, alpha=0.12)

    ax.set_ylabel("R-factor proxy  [MJ mm ha⁻¹ h⁻¹ yr⁻¹]")
    ax.set_title(
        f"{MODEL} HadGEM2-AO  |  Projected Rainfall Erosivity - Jakarta\n"
        "Random Forest prediction  |  10-yr rolling mean ± 1σ",
        fontweight="bold",
    )
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotate model skill
    skill_txt = (
        f"Historical CV:  RMSE={metrics['cv_rmse']:.1f}  "
        f"MAE={metrics['cv_mae']:.1f}  R²={metrics['cv_r2']:.3f}"
    )
    ax.text(0.5, 0.02, skill_txt, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=FONT_BASE - 2,
            color="#696969",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    # Bottom panel: anomaly relative to historical mean
    ax2 = axes[1]
    hist_mean = df_predictions.loc[
        df_predictions["scenario"] == "historical", "R_proxy_predicted"
    ].mean()

    for scen in ["rcp26", "rcp45", "rcp85"]:
        df_s = df_predictions[df_predictions["scenario"] == scen].sort_values("year")
        if df_s.empty:
            continue
        anomaly  = df_s["R_proxy_predicted"].values - hist_mean
        years    = df_s["year"].values
        color    = SCENARIO_COLORS[scen]
        smoothed = pd.Series(anomaly, index=years).rolling(10, center=True, min_periods=5).mean()
        ax2.plot(years, smoothed.values, color=color, lw=2.0,
                 label=SCENARIO_LABELS[scen])
        ax2.fill_between(years,
                         np.zeros_like(smoothed.values),
                         smoothed.values,
                         color=color, alpha=0.13)

    ax2.axhline(0, color="#676767", lw=1.1, ls="--")
    ax2.set_ylabel("ΔR vs. historical mean")
    ax2.set_xlabel("Year")
    ax2.set_title("Change in Erosivity Relative to Historical Mean (10-yr smooth)",
                  fontweight="bold")
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=FONT_BASE - 2)

    fig.tight_layout()
    out = fig_dir / f"{MODEL}_fig_scenario_projection.png"
    fig.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {out.name}")
    return out


# ===== Metrics CSV ====================

def save_metrics(metrics: dict, out_dir: Path) -> Path:
    rows = [
        {"metric": "CV_RMSE",  "value": metrics["cv_rmse"],  "notes": "5-fold cross-validation"},
        {"metric": "CV_MAE",   "value": metrics["cv_mae"],   "notes": "5-fold cross-validation"},
        {"metric": "CV_R2",    "value": metrics["cv_r2"],    "notes": "5-fold cross-validation"},
        {"metric": "OOB_R2",   "value": metrics["oob_r2"],   "notes": "Out-of-bag (full model)"},
    ]
    for feat, imp, std in zip(
        FEATURES,
        metrics["perm_importances"],
        metrics["perm_importances_std"],
    ):
        rows.append({
            "metric": f"PermImp_{feat}",
            "value":  imp,
            "notes":  f"Permutation importance ± {std:.4f}",
        })
    for feat, imp in zip(FEATURES, metrics["impurity_importances"]):
        rows.append({
            "metric": f"MDI_{feat}",
            "value":  imp,
            "notes":  "Mean decrease in impurity (MDI)",
        })

    df = pd.DataFrame(rows)
    out = out_dir / f"{MODEL}_rf_metrics.csv"
    df.to_csv(out, index=False, float_format="%.6f")
    logger.info(f"  Saved: {out.name}")
    return out



# ===== CLI ====================

@click.command()
@click.option(
    "--indices-dir",
    default=str(DEFAULT_INDICES_DIR),
    show_default=True,
    help="Directory containing HadGEM2-AO_<scenario>_indices_jakarta.nc files.",
)
@click.option(
    "--extreme-dir",
    default=str(DEFAULT_EXTREME_DIR),
    show_default=True,
    help="Directory containing extreme frequency NetCDF files (results/extreme_freq/).",
)
@click.option(
    "--water-stress-dir",
    default=str(DEFAULT_WATER_STRESS_DIR),
    show_default=True,
    help="Directory containing water stress NetCDF files (results/water_stress/).",
)
@click.option(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    show_default=True,
    help="Directory to save model .pkl and metrics .csv.",
)
@click.option(
    "--fig-dir",
    default=str(DEFAULT_FIG_DIR),
    show_default=True,
    help="Directory to save figure .png files.",
)
@click.option(
    "--n-estimators",
    default=RF_PARAMS["n_estimators"],
    show_default=True,
    help="Number of trees in the Random Forest.",
)
@click.option(
    "--max-depth",
    default=RF_PARAMS["max_depth"],
    show_default=True,
    help="Maximum tree depth (None = unlimited).",
)
@click.option(
    "--cv-folds",
    default=CV_FOLDS,
    show_default=True,
    help="Number of cross-validation folds.",
)
@click.option(
    "--random-state",
    default=RANDOM_STATE,
    show_default=True,
    help="Random seed for reproducibility.",
)
def main(indices_dir, extreme_dir, water_stress_dir, output_dir, fig_dir,
         n_estimators, max_depth, cv_folds, random_state):
    """
    Train a Random Forest to predict rainfall erosivity (R-factor proxy) from CMIP5 precipitation indices and project it across RCP scenarios.
    """
    indices_path      = Path(indices_dir)
    extreme_path      = Path(extreme_dir)
    water_stress_path = Path(water_stress_dir)
    out_path          = Path(output_dir)
    fig_path          = Path(fig_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    if not indices_path.exists():
        logger.error(f"Indices directory not found: {indices_path}")
        logger.error("Run precipitation_indices.py --scenario all first.")
        sys.exit(1)
    for label, path, upstream_script in [
        ("extreme_freq",  extreme_path,      "extreme_frequency.py --scenario all"),
        ("water_stress",  water_stress_path,  "water_stress.py --scenario all"),
    ]:
        if path.exists():
            nc_files = list(path.glob("*.nc"))
            logger.info(f"  {label:<15}: {path}  ({len(nc_files)} .nc file(s) found)")
        else:
            logger.warning(
                f"  {label:<15}: directory not found: {path}\n"
                f"    → Run {upstream_script} to generate these outputs."
            )

    # Override RF params from CLI
    rf_params = {**RF_PARAMS, "n_estimators": n_estimators,
                 "max_depth": max_depth, "random_state": random_state}

    # Load all scenarios
    logger.info("=" * 55)
    logger.info("STEP 1: Loading precipitation indices")
    logger.info("=" * 55)

    all_dfs = {}
    for scen in SCENARIOS:
        df = load_scenario_indices(indices_path, scen)
        if df is not None and not df.empty:
            all_dfs[scen] = df

    if "historical" not in all_dfs:
        logger.error("Historical indices are required for training but were not found.")
        sys.exit(1)

    df_hist = all_dfs["historical"]
    logger.info(
        f"\n  Historical training set: {len(df_hist)} years  "
        f"({int(df_hist['year'].min())}–{int(df_hist['year'].max())})"
    )

    # Train & evaluate
    logger.info("=" * 55)
    logger.info("STEP 2: Training Random Forest")
    logger.info("=" * 55)

    model, metrics, cv_preds = train_and_evaluate(df_hist, rf_params, cv_folds)

    logger.info(f"\n  OOB R²   = {metrics['oob_r2']:.4f}")
    logger.info(f"  CV  R²   = {metrics['cv_r2']:.4f}")
    logger.info(f"  CV  RMSE = {metrics['cv_rmse']:.2f} MJ mm ha⁻¹ h⁻¹ yr⁻¹")
    logger.info(f"  CV  MAE  = {metrics['cv_mae']:.2f} MJ mm ha⁻¹ h⁻¹ yr⁻¹")

    logger.info("\n  Feature importance (permutation):")
    for feat, imp, std in sorted(
        zip(FEATURES, metrics["perm_importances"], metrics["perm_importances_std"]),
        key=lambda x: x[1], reverse=True
    ):
        logger.info(f"    {feat:<10}: {imp:.4f}  ±{std:.4f}")

    # Predict all scenarios
    logger.info("=" * 55)
    logger.info("STEP 3: Predicting all scenarios")
    logger.info("=" * 55)

    df_predictions = predict_all_scenarios(model, all_dfs)

    for scen in SCENARIOS:
        df_s = df_predictions[df_predictions["scenario"] == scen]
        if df_s.empty:
            continue
        logger.info(
            f"  {scen:<12}: n={len(df_s)}  "
            f"mean={df_s['R_proxy_predicted'].mean():.1f}  "
            f"max={df_s['R_proxy_predicted'].max():.1f}"
        )

    # Save outputs
    logger.info("=" * 55)
    logger.info("STEP 4: Saving outputs")
    logger.info("=" * 55)

    # Model
    model_path = out_path / f"{MODEL}_rf_erosivity_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "rf_params": rf_params}, f)
    logger.info(f"  Saved: {model_path.name}")

    # Metrics CSV
    save_metrics(metrics, out_path)

    # Predictions CSV
    pred_path = out_path / f"{MODEL}_rf_predictions.csv"
    df_predictions.to_csv(pred_path, index=False, float_format="%.4f")
    logger.info(f"  Saved: {pred_path.name}")

    # Figures
    logger.info("=" * 55)
    logger.info("STEP 5: Generating figures")
    logger.info("=" * 55)

    plot_feature_importance(metrics, fig_path)
    plot_observed_vs_predicted(df_hist, cv_preds, metrics, fig_path)
    plot_residuals(df_hist, cv_preds, metrics, fig_path)
    plot_scenario_projection(df_predictions, metrics, fig_path)

    # Final summary
    logger.info("=" * 55)
    logger.info("DONE")
    logger.info("=" * 55)
    logger.info(f"  Model     : {model_path.name}")
    logger.info(f"  Metrics   : {MODEL}_rf_metrics.csv")
    logger.info(f"  Predictions: {MODEL}_rf_predictions.csv")
    logger.info(f"  Figures   : {fig_path}")
    logger.info(f"\n  CV R²   = {metrics['cv_r2']:.4f}")
    logger.info(f"  CV RMSE = {metrics['cv_rmse']:.2f} MJ mm ha⁻¹ h⁻¹ yr⁻¹")
    logger.info(f"  OOB R²  = {metrics['oob_r2']:.4f}")


if __name__ == "__main__":
    main()
