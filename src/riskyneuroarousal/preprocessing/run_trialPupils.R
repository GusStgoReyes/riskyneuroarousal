library(eyelinker)
library(mgcv)
library(dplyr)
library(eyeris)
library(gsignal)
source("load_data.R")
source("remove_gaze_regressor.R")
source("replace_out_of_bounds.R")
source("add_nan_descriptors.R")
source("create_blink_info.R")
source("valid_subs.R")

# Define input and output directories
home_dir <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/PAPERS/
paper1_loss_aversion_pupil/eye_data"
input_dir <- paste(home_dir, "NARPS_MG_asc", sep = "/")
output_dir <- paste(home_dir, "NARPS_MG_asc_processed", sep = "/")
quality_control_dir <- paste(home_dir,
                             "NARPS_MG_asc_quality_control", sep = "/")

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Ensure the quality control directory exists
if (!dir.exists(quality_control_dir)) {
  dir.create(quality_control_dir)
}

# Process all .asc files in the input directory
asc_files <- list.files(input_dir, pattern = "\\.asc$", full.names = TRUE)

for (file in asc_files) {
  # Load the data
  loaded_data <- load_data(file, valid_subs)
  # if loaded_data is null, skip the file
  if (is.null(loaded_data)) {
    next
  }

  # Make the data items available in the global environment
  data <- loaded_data$data
  subject_id <- loaded_data$subject_id
  run_id <- loaded_data$run_id
  subject_num <- loaded_data$subject_num
  run_num <- loaded_data$run_num

  # (2) Regress out gaze position from pupil size
  tryCatch({
    data <- remove_gaze_regressor(data)
  }, error = function(e) {
    print(paste("Error in removing gaze regressor for", subject_id, run_id))
    next
  })

  # (2) Save an image of gaze position
  # filename <- paste0(subject_id, "_", run_id, "_gaze.png")
  # plot_density(data, output_dir = quality_control_dir, filename = filename)

  # (3) Replace out of bounds values with NA
  data <- replace_out_of_bounds(data)

  # (4) Add indicator variables for blink and out of bounds
  data <- add_nan_descriptors(data)

  # (5) Get the mean pupil size (for later use)
  mean_pupil_size <- mean(data$timeseries$pupil_raw, na.rm = TRUE)

  # (6) Filter the data by:
  # I. Deblinking (100 ms before and after blink)
  # II. Removing physiological artifacts
  # III. Interpolating the results
  # IV. Applying a low-pass filter (0.02 Hz)
  eye_preproc <- (eyeris::deblink(data, extend = 100) |>
                    eyeris::detransient(n = 16) |>
                    eyeris::interpolate() |>
                    eyeris::lpfilt(wp = 0.02, ws = 0.04, rp = 1, rs = 35))

  # (7) Bandpass filter the data (0.02-4 Hz)
  # fs <- eye_preproc$info$sample.rate
  # fpass <- c(0.02, 4)
  # wpass <- fpass / (fs / 2)
  # but <- gsignal::butter(2, wpass, "pass")
  # eye_preproc$timeseries$pupil_raw_deblink_detransient_interpolate_lpfilt <-
  #   gsignal::filtfilt(but,
  #                     eye_preproc$timeseries$
  #                       pupil_raw_deblink_detransient_interpolate)
  # + mean_pupil_size

  # (8) Save image of time series of pupil size
  # plot_pupil_series(eye_preproc, output_dir = quality_control_dir, filename = paste0(subject_id, "_", run_id, "_pupil_series.png"))

  # (9) Downsample to 50 Hz
  target_hz <- 50
  original_hz <- eye_preproc$info$sample.rate
  resample_factor <- original_hz / target_hz

  eye_preproc$timeseries <- eye_preproc$timeseries %>%
    mutate(bin = ceiling(row_number() / resample_factor)) %>%
    group_by(bin) %>%
    summarise(
      # Apply mean to specific columns
      eye_x = mean(eye_x, na.rm = TRUE),
      eye_y = mean(eye_y, na.rm = TRUE),


      # Apply median to specific columns
      pupil_raw = median(pupil_raw, na.rm = TRUE),
      pupil_raw_deblink = median(pupil_raw_deblink, na.rm = TRUE),
      pupil_raw_deblink_detransient = median(pupil_raw_deblink_detransient, na.rm = TRUE),
      pupil_raw_deblink_detransient_interpolate = median(pupil_raw_deblink_detransient_interpolate, na.rm = TRUE),
      pupil_raw_deblink_detransient_interpolate_lpfilt = median(pupil_raw_deblink_detransient_interpolate_lpfilt, na.rm = TRUE),

      # Pick a single value (first row in each bin)
      time_orig = first(time_orig),
      eye = first(eye),
      hz = first(hz),
      type = first(type),
      blink = first(blink),
      outofbounds = first(outofbounds),

      .groups = "drop"
    )

  eye_preproc$info$sample.rate <- 50
  eye_preproc$timeseries$hz <- 50

  # (10) Epoch the data and compute baseline pupil size for trials
  eye_preproc <- eyeris::epoch(eye_preproc, events = "flag_TrialStart*", calc_baseline = TRUE, apply_baseline = FALSE, baseline_events = "flag_TrialStart*", baseline_period = c(-0.5, 0), limits = c(0, 4))
  eye_preproc <- eyeris::epoch(eye_preproc, events = "flag_Response*", calc_baseline = FALSE, apply_baseline = FALSE, limits = c(-1.5, 1.5))
  eye_preproc$epoch_flagTrialstart$trial <- as.numeric(gsub(".*_Trial(\\d+)_.*", "\\1", eye_preproc$epoch_flagTrialstart$matched_event))
  eye_preproc$epoch_flagResponse$trial <- as.numeric(gsub(".*_Trial(\\d+)_.*", "\\1", eye_preproc$epoch_flagResponse$matched_event))

  eye_preproc$epoch_flagTrialstart$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagTrialstart$trial]
  eye_preproc$epoch_flagResponse$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagResponse$trial]
  eye_preproc$epoch_flagTrialstart$ps_preprocessed <- (eye_preproc$epoch_flagTrialstart$pupil_raw_deblink_detransient_interpolate_lpfilt - eye_preproc$epoch_flagTrialstart$baseline) / (eye_preproc$epoch_flagTrialstart$baseline + mean_pupil_size)
  eye_preproc$epoch_flagResponse$ps_preprocessed <- (eye_preproc$epoch_flagResponse$pupil_raw_deblink_detransient_interpolate_lpfilt - eye_preproc$epoch_flagResponse$baseline) / (eye_preproc$epoch_flagResponse$baseline + mean_pupil_size)


  # eye_preproc$epoch_flagTrialstart$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagTrialstart$trial]
  # eye_preproc$epoch_flagResponse$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagResponse$trial]
  # eye_preproc$epoch_flagTrialstart$ps_preprocessed <- (eye_preproc$epoch_flagTrialstart$pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly - eye_preproc$epoch_flagTrialstart$baseline) / (eye_preproc$epoch_flagTrialstart$baseline + mean_pupil_size)
  # eye_preproc$epoch_flagResponse$ps_preprocessed <- (eye_preproc$epoch_flagResponse$pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly - eye_preproc$epoch_flagResponse$baseline) / (eye_preproc$epoch_flagResponse$baseline + mean_pupil_size)
  eye_preproc$epoch_flagTrialstart$sub <- subject_num
  eye_preproc$epoch_flagResponse$sub <- subject_num
  eye_preproc$epoch_flagTrialstart$trial <- eye_preproc$epoch_flagTrialstart$trial + (run_num-1) * 64
  eye_preproc$epoch_flagResponse$trial <- eye_preproc$epoch_flagResponse$trial + (run_num-1) * 64

  # (10) Select the relevant columns to save
  output_data <- eye_preproc$epoch_flagTrialstart %>%
      select(sub, timebin, trial, ps_preprocessed, blink, outofbounds)

  output_data2 <- eye_preproc$epoch_flagResponse %>%
      select(sub, timebin, trial, ps_preprocessed, blink, outofbounds)

  # (11) Save output data
  output_filename <- paste0(subject_id, "_", run_id, "_timeseries_start.csv")
  write.csv(output_data, file.path(output_dir, output_filename), row.names = FALSE)

  output_filename2 <- paste0(subject_id, "_", run_id, "_timeseries_response.csv")
  write.csv(output_data2, file.path(output_dir, output_filename2), row.names = FALSE)

  # (12) Create blink info
  # blink_info <- create_blink_info(eye_preproc, subject_num, run_num)
  # blink_output_filename <- paste0(subject_id, "_", run_id, "_blink_info.csv")
  # write.csv(blink_info, file.path(output_dir, blink_output_filename), row.names = FALSE)
}