# erosivity_rf.py

Random Forest model for predicting rainfall erosivity (R-factor) over the
Special Capital Region of Jakarta using CMIP6 precipitation indices.

## Per-model training

One RF is trained per CMIP6 model using that model's QDM-corrected historical indices as features and the Bols (1978) R-proxy as the training target. Each
RF is then applied exclusively to its own model's future projections. Final ensemble projections are produced as a performance-weighted mean and std using
weights from ensemble_validation.py (ensemble_weights.json).

This approach avoids distributional shift between training and inference: the RF learns the index→R relationship as expressed by that specific model's grid and
residual distributional characteristics after QDM, so applying it to that model's future output is internally consistent.

## R-factor proxy (training target)

Bols (1978) Indonesia calibration:

  `R = α × PRCPTOT^(1 - β) × Rx1day^β`
  `R = 6.19 × PRCPTOT^0.24 × Rx1day^0.76`

This is used as the *label* during training. The RF then learns to reproduce
this relationship from the full feature set, capturing nonlinear interactions
among indices that the analytic formula cannot express.

Feature set
-----------
    PRCPTOT     Annual total wet-day precipitation  [mm/yr]
    Rx1day      Annual maximum 1-day precipitation  [mm]
    Rx3day      Annual maximum 3-day precipitation  [mm]
    Rx5day      Annual maximum 5-day precipitation  [mm]
    WDF         Annual wet-day frequency            [days/yr]
    SDII        Simple daily intensity index        [mm/wet-day]
    SPI12_mean  Annual mean of monthly SPI-12       [dimensionless]

SPI12_mean captures the multi-month accumulated moisture deficit signal not
fully expressed by annual totals — physically relevant because prolonged dry
periods followed by intense rainfall (soil crusting → enhanced runoff) are a
key erosivity mechanism in Jakarta's monsoonal regime.

## Outputs

Per model (results/models/):
    random_forest_erosivity_{model}.pkl               trained RF + metadata
    {model}_rf_metrics.csv                            CV + OOB scores, feature importances

Ensemble (results/erosivity/):
    R_bols_{model}_{scenario}_{ensemble}_jakarta.nc   per-model R predictions
    ensemble_rf_predictions.csv                       all models × scenarios

Figures (results/figures/):
    {model}_fig_feature_importance.png
    {model}_fig_observed_vs_predicted.png
    {model}_fig_residuals.png
    fig_ensemble_projection.png                       weighted ensemble + uncertainty band

## References

Bols, P. (1978). The Iso-Erodent Map of Java and Madura. Belgian Technical Assistance Project ATA 105. Soil Research Institute.
Knutti, R., J.Sedláček, B. M.Sanderson, R.Lorenz, E. M.Fischer, and V.Eyring (2017), A climate model projection weighting scheme accounting for performance and interdependence, Geophys. Res. Lett., 44, 1909–1918, doi:10.1002/2016GL072012.


# GloREDa_scaling.py

Post-processing script that anchors Bols (1978) R-factor projections — as produced by erosivity_rf.py — to the absolute magnitude reported by GloREDa (Panagos et al. 2017) for the Jakarta domain.

## Scientific rationale

Daily-scale R-factor proxies (Bols 1978) systematically underestimate EI₃₀-based values because they cannot resolve within-storm intensity peaks.
This bias is:
1. Systematic, consistent across all time periods
2. Proportional, scales with the magnitude of R
3. Well-documented in tropical literature (Vrieling et al. 2010; Panagos et al. 2017)

Because the bias is systematic, the relative climate change signal (future R / historical R) is unaffected. This applies a multiplicative scaling factor derived from GloREDa to correct absolute magnitudes while preserving the model-projected change signal:

`scale_factor = R_GloREDa / R_Bols_historical_mean`
`R_scaled(t)  = R_Bols(t) × scale_factor`

## Outputs

For each model × scenario:

`R_bols_raw_{model}_{scenario}_{ensemble}.nc`       ← copy of RF output, unchanged
`R_gloreda_scaled_{model}_{scenario}_{ensemble}.nc` ← scaled version

## Usage

  ### Fill in GLOREDA_R_JAKARTA before running (see placeholder below)
  `python gloreda_scaling.py`

  ### Override GloREDa value at runtime
  `python gloreda_scaling.py --gloreda-r 9500.0`

  ### Process a single model/scenario
  `python gloreda_scaling.py --model EC-Earth3 --scenario ssp245`

## Input filename convention (from erosivity_rf.py)

  `R_bols_{model}_{scenario}_{ensemble}_jakarta.nc`

Variable names expected inside each file:

`R_bols          : raw Bols R-factor  [MJ·mm·ha⁻¹·h⁻¹·yr⁻¹]`
`R_bols_weighted : weighted ensemble R (present if erosivity_rf.py ran ensemble aggregation), scaled separately if present`