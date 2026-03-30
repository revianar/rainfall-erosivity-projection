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

## References

Bols, P. (1978). The Iso-Erodent Map of Java and Madura. Belgian Technical Assistance Project ATA 105. Soil Research Institute.
Panagos et al. (2017). *Global rainfall erosivity assessment based on high-temporal resolution rainfall records.* Scientific Reports, 7, 4175.