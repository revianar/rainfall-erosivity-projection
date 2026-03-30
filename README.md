# Projected Changes in Rainfall Erosivity Under CMIP6 SSP Scenarios — DKI Jakarta, Indonesia

## Overview

This repository contains the full analysis workflow for projecting future changes in rainfall erosivity (R-factor) over the Special Capital Region of Jakarta (DKI Jakarta), Indonesia, under three Shared Socioeconomic Pathway scenarios using bias-corrected multi-model CMIP6 precipitation simulations.

The study quantifies how climate change may alter the erosive power of rainfall over Jakarta's greater capital region through 2100, with implications for urban soil erosion risk, sediment loading in flood infrastructure, and land degradation in the Jabodetabek area. The R-factor is estimated using the **Bols (1978) Indonesia-specific calibration** and refined through a **Random Forest regression** trained on ETCCDI precipitation indices. Absolute magnitudes are anchored to the **GloREDa global erosivity dataset** via multiplicative scaling.

| | Details |
|---|---|
| **Study region** | DKI Jakarta Greater Capital Region (Jabodetabek) |
| **Domain** | lat [−8.75°, −3.75°], lon [103.125°, 111.875°] |
| **Observed reference** | CHIRPS v2.0 daily (1981–2014) |
| **Bias correction baseline** | 1981–2014 |
| **Historical period** | 1950–2014 |
| **Near-term projection** | 2021–2050 |
| **Far-term projection** | 2071–2100 |
| **Scenarios** | SSP1-2.6, SSP2-4.5, SSP5-8.5 |
| **GCM resolution** | ~1–2° |

## Models

| Model | Institution | Ensemble | Grid | Calendar | Scenarios |
|---|---|---|---|---|---|
| **MRI-ESM2-0** | MRI, Japan | r1i1p1f1 | gn | Gregorian | historical, ssp126, ssp245, ssp585 |
| **EC-Earth3** | EC-Earth Consortium | r1i1p1f1 | gr | Proleptic Gregorian | historical, ssp126, ssp245, ssp585 |
| **CNRM-CM6-1** | CNRM-CERFACS, France | r1i1p1f2 | gr | 360-day | historical, ssp126, ssp245, ssp585 |

All files use the variable `pr` (kg m⁻² s⁻¹), converted to mm/day by multiplying by 86,400.

## Quick Start

1. Install dependencies:
```
pip install xarray numpy pandas matplotlib cartopy scipy scikit-learn netCDF4 rasterio shapely click
conda install -c conda-forge cartopy  # Cartopy may require system-level dependencies
```
2. Download daily CMIP6 `pr` files from [ESGF](https://esgf-node.llnl.gov/search/cmip6/) for the three models above (historical + ssp126/ssp245/ssp585) and place them in `../CMIP6/`
3. Download CHIRPS v2.0 daily files (1981–2014) into `../CHIRPS/`
4. Download GloREDa `GlobalR_NoPol.tif` from [JRC ESDAC](https://esdac.jrc.ec.europa.eu/content/global-rainfall-erosivity) into `../GloREDa/`
5. Run scripts in order (see **Scripts** section below)
6. Run notebooks in order (see **Notebooks** section below)

---

## Methodology

### 1. Domain Cropping
Raw CMIP6 yearly chunks are merged, cropped to the Jakarta bbox, and converted to mm/day by `crop_domain.py`. Each model's chunks are discovered via glob on the standard CMIP6 filename convention and merged with `open_mfdataset`. The 0–360° longitude convention used by some models is detected and corrected automatically.

### 2. Quantile Delta Mapping (QDM)
Bias correction follows **Cannon et al. (2015)** QDM, which preserves the relative change signal (climate delta) while correcting distributional bias against observations.

- **Calibration period:** 1981–2014 against CHIRPS v2.0
- **Quantile bins:** 100
- **Wet-day threshold:** ≥ 1 mm/day
- CHIRPS is regridded to each model's native grid via nearest-neighbour interpolation before fitting
- Dry-day frequency bias is corrected first: excess model wet days are demoted to dry in order of ascending intensity
- Intensity correction applies a **multiplicative delta** per quantile: δ = x_future / Q_hist(τ), preserving the future change signal relative to the historical CDF
- The delta is capped at 5× to prevent runaway extrapolation beyond the observed range
- Ocean cells are infilled from the nearest land cell before fitting to avoid NaN contamination

```
Transfer function (per grid cell):
  τ   = F_hist(x_f)          find quantile in historical CDF
  δ   = x_f / Q_hist(τ)      multiplicative climate delta
  x*  = Q_obs(τ)             map τ onto observed CDF
  x_c = δ × x*               apply delta to observation-mapped value
```

### 3. Ensemble Validation & Weighting
Multi-metric NMAE skill scoring is used to derive performance-based ensemble weights from the pre-QDM historical run against CHIRPS. Five metrics are evaluated at the Jakarta centre cell (lat = −7.5°, lon = 106.875°):

| Metric | Weight | Rationale |
|---|---|---|
| PRCPTOT | 0.35 | Dominant exponent in the Bols R-factor formula |
| Rx1day | 0.20 | Controls extreme tail; directly enters R-factor ratio |
| SDII | 0.15 | Wet-day intensity; second term of the Bols formula |
| WDF | 0.15 | Wet/dry partitioning shapes all intensity metrics |
| Seasonal correlation | 0.15 | Monsoon phase skill; critical for physical plausibility |

Weights are normalised to sum to 1. Post-QDM (BC) weights are all ≈ 0.333 by construction — QDM forces CDF convergence during calibration, making the BC weights uninformative. Downstream aggregation uses the **pre-QDM (`raw`) weights** following Knutti et al. (2017).

### 4. ETCCDI Precipitation Indices
Six indices are computed annually from QDM-corrected daily precipitation using explicit year-loop groupby (cftime-safe; `.resample()` is never used):

$$PRCPTOT = \sum RR_{ij}, \quad RR \geq 1 \text{ mm/day}$$

| Index | Definition | Units |
|---|---|---|
| PRCPTOT | Annual total wet-day precipitation | mm/year |
| Rx1day | Annual maximum 1-day precipitation | mm |
| Rx3day | Annual maximum 3-day accumulated precipitation | mm |
| Rx5day | Annual maximum 5-day accumulated precipitation | mm |
| WDF | Annual wet-day frequency (days with pr ≥ 1 mm/day) | days/year |
| SDII | Simple Daily Intensity Index (PRCPTOT / WDF) | mm/wet-day |

### 5. Water Stress
Potential evapotranspiration (PET) is estimated using the **Hargreaves-Samani** method. CMIP6 tasmax/tasmin files were not downloaded; temperature is constructed from IPCC AR6 WGI Atlas Southeast Asia region warming deltas applied to the BMKG Jakarta climatological baseline (Tmax = 32°C, Tmin = 24°C), with asymmetric warming (Tmax warms 10% faster than Tmin, consistent with observed DTR narrowing in tropical land areas).

Derived metrics: Aridity Index (AI = P/PET), Annual Moisture Deficit (P − PET), SPI-12, and Water-Stressed Months (months where P < 0.5 × PET).

### 6. Extreme Frequency Analysis
GEV distributions are fitted by MLE to Annual Maximum Series (AMS) of Rx1day, Rx3day, and Rx5day at the Jakarta centre cell. Return levels for 2–100 year return periods are computed. Jakarta operational flood thresholds (100 / 150 / 200 mm/day) are assessed for annual exceedance probability change across scenarios.

> **Note on degenerate fits:** CNRM-CM6-1 Rx3day produces GEV shape parameters near ξ = −1 due to extreme outliers in its 360-day calendar AMS, causing physically implausible return levels (> 5,000 mm). These fits are flagged and excluded from the analysis with full transparency.

### 7. Random Forest Erosivity (Interpretation C — Per-Model Training)
A Random Forest regressor is trained **per model** on that model's own QDM-corrected historical indices, with the **Bols (1978) R-proxy** as the training target. Each model's RF is applied exclusively to that model's own future projections, avoiding cross-model contamination of the climate change signal.

$$R = 6.19 \times PRCPTOT^{0.76} \times \left(\frac{Rx1day}{SDII}\right)^{0.1}$$

**Feature set:** PRCPTOT, Rx1day, Rx3day, Rx5day, WDF, SDII, SPI12_mean (if available)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| n_estimators | 300 |
| max_depth | 12 |
| min_samples_split | 5 |
| min_samples_leaf | 3 |
| max_features | sqrt |
| bootstrap | True |
| oob_score | True |
| random_state | 42 |

Validation uses 5-fold cross-validation (CV R², RMSE) and OOB R².

> **Note on the EI₃₀ proxy:** True EI₃₀ requires sub-hourly rainfall intensity, which is not resolvable from daily CMIP6 output. The Bols daily proxy is defensible because the bias relative to GloREDa is systematic and cancels in relative change analysis (Vrieling et al. 2010).

### 8. GloREDa Scaling
The RF R-factor output is anchored to the **GloREDa** observed present-day erosivity (Panagos et al. 2017) via **multiplicative scaling (Option B)**:

`\text{scale\_factor} = \frac{R_{\text{GloREDa}}}{{\overline{R_{\text{Bols, hist}}}}}`

This preserves the relative change signal exactly while giving physically interpretable absolute magnitudes. The historical scaled mean equals GloREDa by construction — it is not an independent validation. The scientifically meaningful output is the **relative change**, not the absolute future magnitude.

---

## Scripts

Run in the following order:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `py/crop_domain/crop_domain.py` | Raw CMIP6 `.nc` chunks | `py/data/processed/pr_day_{model}_{scenario}_{ensemble}_jakarta.nc` |
| 2 | `py/bias_correction/QDM.py prepare` | Processed files + CHIRPS | Copies into `py/data/bias_corrected/` + `chirps_v2_jakarta_1981_2014.nc` |
| 3 | `py/bias_correction/QDM.py apply` | Bias-corrected dir | `pr_day_{model}_{scenario}_{ensemble}_jakarta_qdm.nc` |
| 4 | `py/validation/ensemble_validation.py` | Raw + QDM historical | `py/results/ensemble_weights.json` + `ensemble_validation_metrics.csv` |
| 5 | `py/indices/precipitation_indices.py` | QDM files | `{model}_{scenario}_{ensemble}_indices_jakarta.nc` |
| 6 | `py/indices/water_stress.py` | QDM files | `{model}_{scenario}_{ensemble}_water_stress_{temp_stat}_jakarta.nc` |
| 7 | `py/indices/extreme_frequency.py` | Index files | `{model}_{scenario}_{ensemble}_{var}_extreme_freq_jakarta.nc` |
| 8 | `py/ml/erosivity_rf.py` | Index + water stress files | `R_bols_{model}_{scenario}_{ensemble}_jakarta.nc` + `random_forest_erosivity_{model}.pkl` |
| 9 | `py/postprocessing/GloREDa_scaling.py` | R_bols NetCDFs + GloREDa TIF | `R_gloreda_scaled_{model}_{scenario}_{ensemble}_jakarta.nc` + `gloreda_scale_factors.json` |

All scripts support `--model all --scenario all` CLI flags and skip scenarios not defined for a given model.

**Example:**
```bash
python py/crop_domain/crop_domain.py --model all --scenario all
python py/bias_correction/QDM.py prepare
python py/bias_correction/QDM.py apply --model all --scenario all
python py/validation/ensemble_validation.py
python py/indices/precipitation_indices.py --model all --scenario all
python py/indices/water_stress.py --model all --scenario all --temp-stat median
python py/indices/extreme_frequency.py --model all --scenario all
python py/ml/erosivity_rf.py --model all --scenario all
python py/postprocessing/GloREDa_scaling.py --model all --scenario all
```

---

## Notebooks

| Notebook | Description |
|---|---|
| `py/notebooks/01_data_exploration.ipynb` | Raw CMIP6 file inspection, spatial maps, annual cycle vs CHIRPS, flood threshold exceedances |
| `py/notebooks/QDM.ipynb` | Pre-correction bias assessment, QDM transfer function demo, post-correction QQ-plots and annual cycle validation |
| `py/notebooks/03_precipitation_indices.ipynb` | ETCCDI index spatial maps, time series at Jakarta centre cell, change signal maps, period-mean summary tables |
| `py/notebooks/04_water_stress.ipynb` | Aridity Index maps, SPI-12 time series, water-stressed months, moisture deficit change maps |
| `py/notebooks/05_extreme_frequency.ipynb` | GEV fitting, QQ-plots, return level curves, flood threshold exceedance probabilities, spatial return level maps |
| `py/notebooks/06_erosivity_ml.ipynb` | RF training metrics, feature importances, GloREDa-scaled R-factor time series, weighted ensemble projections, headline % change results |

Kernel: `Python (jakarta-rainfall)` (see Requirements below).

---

## Results Summary

Ensemble weights (pre-QDM dynamical skill):

| Model | Weight | Notes |
|---|---|---|
| MRI-ESM2-0 | 0.2925 | Weakest PRCPTOT skill; poor seasonal correlation |
| EC-Earth3 | 0.3623 | Best PRCPTOT and seasonal correlation skill |
| CNRM-CM6-1 | 0.3452 | Best Rx1day skill of the three |

> All CMIP6 models underestimate PRCPTOT and Rx1day relative to CHIRPS. This is a known bias in maritime Southeast Asia driven by convective parameterisation at coarse resolution.

Headline R-factor change results (GloREDa-scaled, spatially averaged, far-term 2071–2100 vs historical 1950–2014):

| Scenario | Near-term Δ% | Far-term Δ% |
|---|---|---|
| SSP1-2.6 | See notebook 06 | See notebook 06 |
| SSP2-4.5 | See notebook 06 | See notebook 06 |
| SSP5-8.5 | See notebook 06 | See notebook 06 |

*Run notebook 06 to populate the headline results after executing all scripts.*

---

## File & Directory Structure

```
> Jupyter Notebook/                        ← DATA_ROOT
  > CMIP6/                                 ← Raw CMIP6 NetCDF files
    > pr_day_{model}_{scenario}_{ensemble}_{grid}_{start}-{end}.nc
> CHIRPS/                                  ← Raw CHIRPS v2.0 daily files
> GloREDa/
  > GlobalR_NoPol.tif                      ← GloREDa global erosivity raster
    > Rainfall-Erosivity/                  ← PROJECT_ROOT
      > py/
        > crop_domain/
          > crop_domain.py
        > bias_correction/
          > QDM.py
        > validation/
          > ensemble_validation.py
        > indices/
          > precipitation_indices.py
          > water_stress.py
          > extreme_frequency.py
        > ml/
          > erosivity_rf.py
        > postprocessing/
          > GloREDa_scaling.py
        > data/
          > processed/                     ← crop_domain.py output
          > bias_corrected/                ← QDM.py output + CHIRPS file
        > results/
          > indices/                       ← ETCCDI index NetCDFs
          > water_stress/                  ← Water stress NetCDFs
          > extreme_freq/                  ← GEV return level NetCDFs
          > models/                        ← RF .pkl files
          > erosivity/                     ← R_bols NetCDFs
          > erosivity_scaled/
            > raw/                         ← Unscaled R_bols copies
            > scaled/                      ← GloREDa-scaled R_gloreda NetCDFs
          > figures/
          > tables/
            > ensemble_weights.json
            > gloreda_scale_factors.json
        > notebooks/
          > 01_data_exploration.ipynb
          > 02_QDM.ipynb
          > 03_precipitation_indices.ipynb
          > 04_water_stress.ipynb
         > 05_extreme_frequency.ipynb
          > 06_erosivity_ml.ipynb
      > README.md
```

---

## Requirements

```
python>=3.9
xarray
numpy
pandas
matplotlib
cartopy
scipy
scikit-learn
netCDF4
rasterio
shapely
click
```

Install via conda (recommended for Cartopy and rasterio):
```bash
conda create -n rainfall-erosivity python=3.10
conda activate rainfall-erosivity
conda install -c conda-forge xarray numpy pandas matplotlib cartopy scipy scikit-learn netcdf4 rasterio shapely click
```

---

## Outputs

### Figures (150 dpi PNG)
| File | Description |
|---|---|
| `01_mean_annual_precip_{model}.png` | Mean annual precipitation maps per model (historical) |
| `01_annual_cycle_chirps_vs_models.png` | Annual cycle — CHIRPS vs raw CMIP6 |
| `02_annual_cycle_preQDM.png` | Pre-QDM annual cycle bias |
| `02_qq_after_QDM.png` | QQ-plots post-bias correction per model |
| `03_{var}_{model}_historical_mean.png` | ETCCDI index spatial maps |
| `03_indices_timeseries_all.png` | Index time series — all models × scenarios |
| `03_{var}_change_signal_ensemble.png` | Ensemble-mean % change maps per index |
| `04_AI_{model}_historical.png` | Aridity Index maps |
| `04_spi12_timeseries.png` | SPI-12 time series |
| `04_water_stress_months.png` | Water-stressed months projection |
| `04_moisture_deficit_change.png` | Moisture deficit change maps |
| `05_gev_qqplot_historical.png` | GEV goodness-of-fit QQ-plots |
| `05_return_levels_{var}.png` | Return level curves |
| `05_flood_exceedance_bars.png` | Flood threshold exceedance probabilities |
| `06_erosivity_ensemble_timeseries.png` | R-factor ensemble time series |
| `06_erosivity_per_model_ssp585.png` | Per-model vs ensemble comparison |
| `06_erosivity_change_spatial.png` | Spatial R-factor % change maps |
| `06_rf_feature_importance.png` | RF permutation feature importances |

### NetCDF outputs
| File | Description |
|---|---|
| `pr_day_{model}_{scenario}_{ensemble}_jakarta.nc` | Cropped raw precipitation |
| `pr_day_{model}_{scenario}_{ensemble}_jakarta_qdm.nc` | QDM bias-corrected precipitation |
| `{model}_{scenario}_{ensemble}_indices_jakarta.nc` | Annual ETCCDI indices |
| `{model}_{scenario}_{ensemble}_water_stress_{stat}_jakarta.nc` | Water stress metrics |
| `{model}_{scenario}_{ensemble}_{var}_extreme_freq_jakarta.nc` | GEV return levels |
| `R_bols_{model}_{scenario}_{ensemble}_jakarta.nc` | Raw RF-predicted R-factor |
| `R_gloreda_scaled_{model}_{scenario}_{ensemble}_jakarta.nc` | GloREDa-scaled R-factor |

### JSON & CSV
| File | Description |
|---|---|
| `ensemble_weights.json` | Pre-QDM and post-QDM model weights |
| `gloreda_scale_factors.json` | Per-model GloREDa scale factors |
| `ensemble_validation_metrics.csv` | Skill scores per model × metric |
| `indices_period_means.csv` | Spatially averaged period-mean ETCCDI indices |
| `indices_pct_change.csv` | % change per model × scenario × index |
| `water_stress_period_means.csv` | Water stress period means |
| `flood_exceedance_probabilities.csv` | GEV flood threshold exceedance by scenario |
| `erosivity_change_signal.csv` | Headline R-factor % change table |

---

## Scope of Interpretation

This study quantifies climate-driven changes in rainfall erosivity and discusses implications for urban flood-sediment risk and soil erosion potential in the Jakarta region through literature-based hydroclimatic interpretation.

It does **not** explicitly model:

- river discharge or streamflow routing
- sediment transport or deposition
- land use change interactions
- population or infrastructure exposure

---

## Known Limitations

- **Coarse spatial resolution:** CMIP6 grids (~1–2°) provide at most 5×5 cells over the Jakarta domain. Sub-regional spatial gradients from the coast to the Puncak highlands are not resolved. Statistical downscaling to a finer grid would substantially improve spatial specificity.
- **EI₃₀ proxy:** True erosivity requires sub-hourly rainfall intensity. The Bols (1978) daily proxy introduces a systematic bias relative to GloREDa that is corrected by multiplicative scaling; relative change signals are preserved exactly.
- **Temperature fallback:** No CMIP6 tasmax/tasmin files were downloaded. PET and water stress metrics use IPCC AR6 SEA region warming deltas applied to a spatially uniform baseline, removing any spatial gradient in temperature forcing across the domain.
- **Three-model ensemble:** Inter-model spread is estimated from three models. The uncertainty envelope is sensitive to individual model outliers, particularly CNRM-CM6-1 which shows degenerate GEV behaviour for Rx3day.
- **No independent spatial validation:** QDM validation is against CHIRPS (the calibration reference); GloREDa comparison is not independent (the scale factor was set to match it). An independent withheld period or station-based validation would strengthen reliability claims.

---

## Citation

- **CHIRPS:** Funk et al. (2015). The climate hazards infrared precipitation with stations — a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066. https://doi.org/10.1038/sdata.2015.66
- **GloREDa:** Panagos et al. (2017). Global rainfall erosivity assessment based on high-temporal resolution rainfall records. *Scientific Reports*, 7, 4175. https://doi.org/10.1038/s41598-017-04282-8
- **Bols (1978):** Bols, P.L. (1978). *The iso-erodent map of Java and Madura*. Belgian Technical Assistance Project ATA 105, Soil Research Institute, Bogor.
- **QDM:** Cannon, A.J., Sobie, S.R., & Murdock, T.Q. (2015). Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes? *Journal of Climate*, 28(17), 6938–6959. https://doi.org/10.1175/JCLI-D-14-00754.1
- **Ensemble weighting:** Knutti, R., Sedláček, J., Sanderson, B.M., Lorenz, R., Fischer, E.M., & Eyring, V. (2017). A climate model projection weighting scheme accounting for performance and interdependence. *Geophysical Research Letters*, 44(4), 1909–1918. https://doi.org/10.1002/2016GL072012
- **MRI-ESM2-0:** Yukimoto et al. (2019). The Meteorological Research Institute Earth System Model Version 2.0, MRI-ESM2.0: Description and Basic Evaluation of the Physical Component. *Journal of the Meteorological Society of Japan*, 97(5), 931–965. https://doi.org/10.2151/jmsj.2019-051
- **EC-Earth3:** Döscher et al. (2022). The EC-Earth3 Earth system model for the Coupled Model Intercomparison Project 6. *Geoscientific Model Development*, 15, 2973–3020. https://doi.org/10.5194/gmd-15-2973-2022
- **CNRM-CM6-1:** Voldoire et al. (2019). Evaluation of CMIP6 DECK Experiments With CNRM-CM6-1. *Journal of Advances in Modeling Earth Systems*, 11(7), 2177–2213. https://doi.org/10.1029/2019MS001683
- **Vrieling et al. (2010):** Vrieling, A., Sterk, G., & de Jong, S.M. (2010). Satellite-based estimation of rainfall erosivity for Africa. *Journal of Hydrology*, 395(3–4), 235–241. https://doi.org/10.1016/j.jhydrol.2010.10.035

---

## Author Note

This repository is designed as a reproducible hydroclimatic research workflow and serves as a self-made portfolio piece. The full pipeline — from raw CMIP6 data to GloREDa-scaled erosivity projections — is implemented as modular, CLI-driven Python scripts with Jupyter notebooks for analysis and visualisation.
