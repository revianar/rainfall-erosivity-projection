## quantile_mapping.py  —  CMIP6 / QDM edition
------------

Bias-corrects CMIP6 daily precipitation using Quantile Delta Mapping (QDM).

### QDM overview

I decided to use QDM because it preserves the *relative change* (delta) between
the historical model and the future model at each quantile, then applies that
delta on top of the observed distribution.  Unlike empirical QM (eQM) which
directly maps future quantiles to the observed distribution, QDM prevents
the well-known eQM artefact of washing out long-term climate trends.

Algorithm (Cannon et al., 2015):
  For each wet-day value x_f in the future scenario:
    1. Find its quantile τ in the historical model CDF:  τ = F_hist(x_f)
    2. Compute the relative delta:  δ = x_f / Q_hist(τ)   (multiplicative)
    3. Map τ onto the observed CDF:  x_obs = Q_obs(τ)
    4. Corrected value:              x_corr = δ × x_obs

  Dry-day frequency is corrected separately using the ratio of wet-day
  probabilities between obs and historical model.

### Per-model design

All functions accept an explicit `model` argument.  The MODELS registry
(copied from crop_domain.py) provides the ensemble and grid labels used
in filenames.  The `apply` subcommand loops over models independently so
each model gets its own transfer functions fitted from its own historical
simulation against CHIRPS.

### Usage

  # 1. Merge CHIRPS + copy cropped model files
  python quantile_mapping.py prepare

  # 2. Apply QDM for all models and all scenarios
  python quantile_mapping.py apply --model all --scenario all

  # 3. Or target a single model / scenario
  python quantile_mapping.py apply --model MRI-ESM2-0 --scenario ssp245

### Reference

  Cannon, A.J., Sobie, S.R., Murdock, T.Q. (2015). Bias Correction of GCM
  Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes
  in Quantiles and Extremes? Journal of Climate, 28(17), 6938-6959.