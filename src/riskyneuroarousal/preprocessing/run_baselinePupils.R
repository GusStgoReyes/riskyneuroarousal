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
home_dir <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/PAPERS/paper1_loss_aversion_pupil/eye_data"
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
                    eyeris::lpfilt(wp = 0.02, ws = 0.04, rp = 1, rs = 35) |>
                    eyeris::zscore())

  # (10) Epoch the data and compute baseline pupil size for trials
  eye_preproc <- eyeris::epoch(eye_preproc, events = "flag_TrialStart*", calc_baseline = TRUE, apply_baseline = FALSE, baseline_events = "flag_TrialStart*", baseline_period = c(-0.5, 0), limits = c(0, 4))

  # (11) Create a list with baseline, trial number, and subject number
  baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_z_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch
  trial_num <- 1:length(baseline) + (run_num - 1) * 64
  sub <- rep(subject_num, length(baseline))
  run <- rep(run_num, length(baseline))

  # (12) Create a data frame with the baseline, trial number, and subject number
  baseline_df <- data.frame(
    baseline = baseline,
    trial_num = trial_num,
    sub = sub,
    run = run
  )

  # (13) Save the baseline data frame to a CSV file
  baseline_filename <- paste0(subject_id, "_", run_id, "_baseline.csv")
  write.csv(baseline_df, file.path(output_dir, baseline_filename), row.names = FALSE)

}