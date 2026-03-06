# Compute all indices from a daily precipitation data.frame
compute_indices_ts <- function(df, threshold = WET_DAY_THRESHOLD) {
  stopifnot("date" %in% names(df), "pr" %in% names(df))

  df <- df %>%
    mutate(
      date  = as.Date(date),
      year  = year(date),
      pr    = pmax(pr, 0),
      is_wet = pr >= threshold
    )

  df %>%
    group_by(year) %>%
    summarise(
      PRCPTOT = sum(pr[is_wet], na.rm = TRUE),
      Rx1day  = compute_rx1day(pr),
      Rx3day  = compute_rxnday(pr, n = 3),
      Rx5day  = compute_rxnday(pr, n = 5),
      WDF     = sum(is_wet, na.rm = TRUE),
      SDII    = ifelse(WDF > 0, PRCPTOT / WDF, NA_real_),
      .groups = "drop"
    )
}

# Annual maximum 1-day precipitation
compute_rx1day <- function(pr) {
  max(pr, na.rm = TRUE)
}

# Annual maximum N-day accumulated precipitation
compute_rxnday <- function(pr, n = 3) {
  if (all(is.na(pr))) return(NA_real_)
  rolling <- zoo::rollsum(pr, k = n, fill = NA, align = "right")
  max(rolling, na.rm = TRUE)
}
