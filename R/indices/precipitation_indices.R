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

# ===== Raster interface (SpatRaster — pixel-by-pixel) ====================

# Compute annual indices from a terra SpatRaster of daily precipitation
compute_indices_raster <- function(r, threshold = WET_DAY_THRESHOLD, years = NULL) {

  dates <- as.Date(terra::time(r))
  yr    <- lubridate::year(dates)

  if (is.null(years)) years <- sort(unique(yr))

  message(sprintf("Computing indices for %d years over %d×%d grid...",
                  length(years), nrow(r), ncol(r)))

  # Pre-allocate output matrices: rows = pixels, cols = years
  n_cells <- terra::ncell(r)
  n_years <- length(years)

  mat <- list(
    PRCPTOT = matrix(NA_real_, n_cells, n_years),
    Rx1day  = matrix(NA_real_, n_cells, n_years),
    Rx3day  = matrix(NA_real_, n_cells, n_years),
    Rx5day  = matrix(NA_real_, n_cells, n_years),
    WDF     = matrix(NA_real_, n_cells, n_years),
    SDII    = matrix(NA_real_, n_cells, n_years)
  )

  for (i in seq_along(years)) {
    y       <- years[i]
    idx     <- which(yr == y)
    r_year  <- r[[idx]]
    vals    <- terra::values(r_year)

    vals[!is.finite(vals) | vals < 0] <- 0

    wet_mask <- vals >= threshold

    mat$PRCPTOT[, i] <- rowSums(vals * wet_mask, na.rm = TRUE)

    mat$Rx1day[, i] <- apply(vals, 1, max, na.rm = TRUE)

    mat$Rx3day[, i] <- apply(vals, 1, compute_rxnday, n = 3)
    mat$Rx5day[, i] <- apply(vals, 1, compute_rxnday, n = 5)

    mat$WDF[, i] <- rowSums(wet_mask, na.rm = TRUE)

    mat$SDII[, i] <- ifelse(
      mat$WDF[, i] > 0,
      mat$PRCPTOT[, i] / mat$WDF[, i],
      NA_real_
    )

    if (i %% 10 == 0 || i == n_years)
      message(sprintf("  Year %d/%d (%d) done", i, n_years, y))
  }

  # Convert matrices back to SpatRasters
  template <- r[[1]]   # geometry template

  rasters <- lapply(names(mat), function(nm) {
    r_out <- terra::rast(
      lapply(seq_len(n_years), function(i) {
        lyr <- template
        terra::values(lyr) <- mat[[nm]][, i]
        lyr
      })
    )
    names(r_out) <- as.character(years)
    terra::time(r_out) <- years
    r_out
  })
  names(rasters) <- names(mat)

  message("All indices computed ✓")
  return(rasters)
}

# ===== Period mean & change signal ====================

# Compute temporal mean of an index raster over a period
period_mean_raster <- function(r_index, period) {
  yrs  <- as.integer(names(r_index))
  idx  <- which(yrs >= period[1] & yrs <= period[2])
  if (length(idx) == 0) stop(paste("No years in period", period[1], "–", period[2]))
  terra::mean(r_index[[idx]], na.rm = TRUE)
}

# Compute change signal between two period means
change_signal <- function(r_future, r_historical, method = "absolute") {
  if (method == "absolute") {
    delta <- r_future - r_historical
  } else if (method == "relative") {
    delta <- ((r_future - r_historical) / r_historical) * 100
  } else {
    stop("method must be 'absolute' or 'relative'")
  }
  return(delta)
}

# ===== Save / load helpers ====================

# Save a named list of index SpatRasters to a single NetCDF file
save_indices_nc <- function(indices_list, outfile, scenario = "") {
  dir.create(dirname(outfile), recursive = TRUE, showWarnings = FALSE)

  # Write each index as a separate variable using ncdf4
  # Use the first raster to get spatial info
  template <- indices_list[[1]]
  years    <- as.integer(names(template))

  lon_vals <- terra::xFromCol(template, 1:ncol(template))
  lat_vals <- terra::yFromRow(template, 1:nrow(template))

  lon_dim <- ncdim_def("lon",  "degrees_east",  lon_vals)
  lat_dim <- ncdim_def("lat",  "degrees_north", lat_vals)
  yr_dim  <- ncdim_def("year", "years",         years, unlim = TRUE)

  var_meta <- list(
    PRCPTOT = list(units = "mm/year",    long = "Annual Total Wet-Day Precipitation"),
    Rx1day  = list(units = "mm",         long = "Annual Maximum 1-Day Precipitation"),
    Rx3day  = list(units = "mm",         long = "Annual Maximum 3-Day Precipitation"),
    Rx5day  = list(units = "mm",         long = "Annual Maximum 5-Day Precipitation"),
    WDF     = list(units = "days/year",  long = "Annual Wet-Day Frequency"),
    SDII    = list(units = "mm/wet-day", long = "Simple Daily Intensity Index")
  )

  nc_vars <- lapply(names(indices_list), function(nm) {
    ncvar_def(nm, var_meta[[nm]]$units,
              list(lon_dim, lat_dim, yr_dim),
              missval = -9999,
              longname = var_meta[[nm]]$long,
              prec = "float")
  })

  nc <- nc_create(outfile, nc_vars)

  for (nm in names(indices_list)) {
    r  <- indices_list[[nm]]
    arr <- array(
      terra::values(r),
      dim = c(ncol(r), nrow(r), nlyr(r))
    )
    ncvar_put(nc, nm, arr)
  }

  ncatt_put(nc, 0, "title",    "ETCCDI Precipitation Indices — Jakarta Greater Capital Region")
  ncatt_put(nc, 0, "model",    "HadGEM2-AO")
  ncatt_put(nc, 0, "scenario", scenario)
  ncatt_put(nc, 0, "created",  as.character(Sys.time()))

  nc_close(nc)
  message("Saved: ", outfile)
}

# Load saved indices NetCDF back into a named list of SpatRasters
load_indices_nc <- function(path) {
  var_names <- c("PRCPTOT", "Rx1day", "Rx3day", "Rx5day", "WDF", "SDII")
  out <- lapply(var_names, function(nm) {
    terra::rast(path, subds = nm)
  })
  names(out) <- var_names
  return(out)
}

message("precipitation_indices.R loaded ✓")
