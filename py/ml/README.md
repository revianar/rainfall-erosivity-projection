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