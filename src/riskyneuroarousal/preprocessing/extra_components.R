# generic handler/wrapper for eyeris pupil pipeline funcs
pipeline_handler2 <- function(eyeris, operation, new_suffix, ...) {
  call_stack <- sys.calls()[[1]]
  eyeris$params[[new_suffix]] <- call_stack

  # getters
  prev_operation <- eyeris$latest
  data <- eyeris$timeseries
  # setters
  output_col <- paste0(prev_operation, "_", new_suffix)
  # run operation
  data[[output_col]] <- operation(data, prev_operation, ...)
  # update S3 eyeris class
  eyeris$timeseries <- data
  # update log var with latest op
  eyeris$latest <- output_col
  return(eyeris)
}

detrend_poly <- function(eyeris, order = 3) {
  return(pipeline_handler2(eyeris, detrend_pupil_poly, "detrend_poly", order))
}

detrend_pupil_poly <- function(x, prev_op, order) {
  pupil <- x[[prev_op]]
  timeseries <- x[["time_orig"]]
  fit <- lm(pupil ~ poly(timeseries, order, raw = TRUE))
  residuals <- fit$residuals
  return(residuals)
}

create_ps_preprocessed <- function(trial, pupil_raw, baseline_means_by_epoch) {
  # Ensure inputs are of the same length
  if (length(trial) != length(pupil_raw) || length(unique(trial)) != length(baseline_means_by_epoch)) {
    stop("Mismatch in lengths of trial, pupil_raw, or baseline_means_by_epoch")
  }
  
  # Calculate ps_preprocessed for each trial
  ps_preprocessed <- sapply(seq_along(trial), function(i) {
    mean_pupil_size <- pupil_raw[i]
    trial_number <- trial[i]
    baseline_mean <- baseline_means_by_epoch[trial_number]
    
    (mean_pupil_size + baseline_mean - baseline_mean) / baseline_mean
  })
  return(ps_preprocessed)
}