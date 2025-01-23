library(eyelinker)
library(mgcv)
library(dplyr)
library(eyeris)
library(ggplot2)

# Define functions for processing pupil data
remove_gaze_regressor <- function(data) {
  # Step 1: Demean the pupil size, excluding NAs
  mean_ps <- mean(data$timeseries$pupil_raw, na.rm = TRUE)  # Compute mean, ignoring NAs
  data$timeseries$pupil_raw_demeaned <- data$timeseries$pupil_raw - mean_ps  # Subtract mean, ignoring NAs
  
  # Step 2: Run the model (bam) to regress out the eye position
  m1 <- bam(
    pupil_raw_demeaned ~ s(eye_x, eye_y, bs = "tp", k = 50),
    data = data$timeseries,
    method = "fREML",
    discrete = TRUE,
    na.action = na.exclude
  )
  
  # Step 3: Get residuals and add the mean back
  residuals_ps <- resid(m1)
  data$timeseries$pupil_raw <- residuals_ps + mean_ps
  
  # Step 4: Remove the pupil_raw_demeaned column
  data$timeseries <- data$timeseries %>%
    select(-pupil_raw_demeaned)

  # Step 5: Return the processed dataset
  return(data)
}

# Replace any pupil_size values that are out of bounds to NaN
replace_out_of_bounds <- function(data, screen_width = 1920, screen_height = 1080, focus_screen_width = 1000, focus_screen_height = 500) {
  data$timeseries$pupil_raw[data$timeseries$eye_x < screen_width/2 - focus_screen_width/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_x > screen_width/2 + focus_screen_width/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_y < screen_height/2 - focus_screen_height/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_y > screen_height/2 + focus_screen_height/2] <- NA
  return(data)
}

add_nan_descriptors <- function(data, screen_width = 1920, screen_height = 1080, focus_screen_width = 1000, focus_screen_height = 500) {
    # Add indicator variable for blink
    data$timeseries$blink <- is.na(data$timeseries$pupil_raw)
    # Add indicator variable for out of bounds
    data$timeseries$outofbounds <- (data$timeseries$eye_x < screen_width/2 - focus_screen_width/2 | 
                                    data$timeseries$eye_x > screen_width/2 + focus_screen_width/2 | 
                                    data$timeseries$eye_y < screen_height/2 - focus_screen_height/2 | 
                                    data$timeseries$eye_y > screen_height/2 + focus_screen_height/2)
                                    
    return(data)
}

# Function to create and save a density plot of the time spent at locations
plot_density <- function(data, output_dir = ".", filename = "density_plot.png") {
  # Compute normalized density
  p <- ggplot(data$timeseries, aes(x = eye_x, y = eye_y)) +
    stat_bin_2d(
      bins = 100,
      aes(fill = after_stat(count / max(count)))  # Normalize density
    ) +
    scale_fill_gradient(low = "blue", high = "red") +  # Color gradient for density
    theme_minimal() +
    labs(
      x = "X Position",
      y = "Y Position",
      fill = "Normalized Density"
    ) +
    xlim(0, 1920) +  # Set x-axis limits
    ylim(0, 1080) +  # Set y-axis limits
    coord_fixed()  # Ensures x and y scales are equal
  
  # Save the plot
  ggsave(filename, plot = p, width = 8, height = 4.5, units = "in", path = output_dir)
}

plot_pupil_series <- function(data, output_dir = ".", filename = "pupil_series_plot.png") {
  # Convert time to seconds
  data$timeseries$time <- (data$timeseries$time_orig - min(data$timeseries$time_orig)) / 1000
  p <- ggplot(data$timeseries, aes(x = time, y = pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly)) +
    geom_line(color = "blue") +
    labs(
      x = "Time (seconds)",
      y = "A.U. Change from Mean"
    ) +
    theme_minimal()

  # Save the plot
  ggsave(filename, plot = p, width = 12, height = 4, units = "in", path = output_dir)
}

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

create_blink_info <- function(eye_preproc, sub_num, run_num) {
  # Create the trial data frame with start times and durations
  blink_info <- eye_preproc$events %>%
    filter(grepl("flag_TrialStart", text)) %>%
    mutate(trial = row_number(),  # Add a trial number
          next_time = lead(time, default = max(eye_preproc$timeseries$time_orig))) %>% 
    select(block, trial, time, next_time) %>%
    rename(trial_start = time)

  # Calculate the number of blinks per trial
  trial_blink_summary <- blink_info %>%
    rowwise() %>%
    mutate(
      num_blinks = sum(eye_preproc$blink$stime >= trial_start & 
                      eye_preproc$blink$stime < next_time & 
                      eye_preproc$blink$block == block),
      duration = next_time - trial_start
    ) %>%
    ungroup() %>%
    select(trial, num_blinks, duration)

  trial_blink_summary$trial <- trial_blink_summary$trial + (run_num-1) * 64
  trial_blink_summary$sub <- sub_num

  return(trial_blink_summary)
}


# Define input and output directories
input_dir <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/Vagus Nerve and Cognition/eyeNARPS/NARPS_MG_asc"
output_dir <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/Vagus Nerve and Cognition/eyeNARPS/NARPS_MG_asc_processed"
quality_control_dir <- "/Users/gustxsr/Documents/Stanford/PoldrackLab/Vagus Nerve and Cognition/eyeNARPS/NARPS_MG_asc_quality_control"

valid_subs <- c(
        "sub-003",
        "sub-004",
        "sub-005",
        "sub-006",
        "sub-009",
        "sub-010",
        "sub-011",
        "sub-014",
        "sub-019",
        "sub-020",
        "sub-022",
        "sub-025",
        "sub-030",
        "sub-033",
        "sub-036",
        "sub-039",
        "sub-040",
        "sub-043",
        "sub-045",
        "sub-047",
        "sub-049",
        "sub-052",
        "sub-053",
        "sub-054",
        "sub-058",
        "sub-060",
        "sub-061",
        "sub-062",
        "sub-063",
        "sub-064",
        "sub-066",
        "sub-070",
        "sub-071",
        "sub-074",
        "sub-075",
        "sub-076",
        "sub-079",
        "sub-080",
        "sub-081",
        "sub-082",
        "sub-084",
        "sub-085",
        "sub-087",
        "sub-089",
        "sub-094",
        "sub-095",
        "sub-098",
        "sub-099",
        "sub-102",
        "sub-105",
        "sub-109",
        "sub-115",
        "sub-118",
        "sub-123",
        "sub-124")

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
    # Extract subject_id and run_id
    subject_id <- stringr::str_extract(file, "sub-\\d{3}")
    run_id <- stringr::str_extract(file, "run-\\d")
    subject_num <- as.numeric(stringr::str_extract(subject_id, "\\d{3}"))
    run_num <- as.numeric(stringr::str_extract(run_id, "\\d"))
    print(paste("Processing", subject_id, run_id))
     # Only do processing for valid subjects
    if (!(subject_id %in% valid_subs)) {
        cat("Skipping subject ", subject_id, " as it is not in the valid list.\n")
        next
    }

    # (1) Load the data (blinks and pupil size)
    data <- eyeris::load_asc(file)
    
    # (2) Save an image of gaze position
    filename <- paste0(subject_id, "_", run_id, "_gaze.png")
    plot_density(data, output_dir = quality_control_dir, filename = filename)

    # (3) Replace out of bounds values with NA
    data <- replace_out_of_bounds(data)

    # (4) Add indicator variables for blink and out of bounds
    data <- add_nan_descriptors(data)

    # (5) Regress out gaze position from pupil size
    tryCatch({
        data <- remove_gaze_regressor(data)
    }, error = function(e) {
        print(paste("Error in removing gaze regressor for", subject_id, run_id))
        next
    })

    # (6) Get the mean pupil size (for later use)
    mean_pupil_size <- mean(data$timeseries$pupil_raw, na.rm = TRUE)

    # (7) Filter the data by deblinking (100 ms before and after blink), removing physiological artifacts
    # interpolating the results, applying a low-pass filter and detrending the data with a cubic polynomial
    eye_preproc <- eyeris::deblink(data, extend = 100) |> eyeris::detransient(n = 16) |> eyeris::interpolate() |> eyeris::lpfilt(wp = 4, ws = 8, rp = 1, rs = 35) |> detrend_poly(order = 3)

    # (8) Save image of time series of pupil size
    plot_pupil_series(eye_preproc, output_dir = quality_control_dir, filename = paste0(subject_id, "_", run_id, "_pupil_series.png"))

    # select every x rows in timeseries to turn to 50hz
    hz <- eye_preproc$info$sample.rate
    resample_factor <- hz / 50
    eye_preproc$timeseries <- eye_preproc$timeseries %>%
      filter(row_number() %% resample_factor == 0)
    eye_preproc$info$sample.rate <- 50
    eye_preproc$timeseries$hz <- 50 

    # (9) Epoch the data and compute baseline pupil size for trials
    eye_preproc <- eyeris::epoch(eye_preproc, events = "flag_TrialStart*", calc_baseline = TRUE, apply_baseline = FALSE, baseline_events = "flag_TrialStart*", baseline_period = c(-0.5, 0), limits = c(0, 4))
    eye_preproc <- eyeris::epoch(eye_preproc, events = "flag_Response*", calc_baseline = FALSE, apply_baseline = FALSE, limits = c(-1.5, 1.5))
    eye_preproc$epoch_flagTrialstart$trial <- as.numeric(gsub(".*_Trial(\\d+)_.*", "\\1", eye_preproc$epoch_flagTrialstart$matched_event))
    eye_preproc$epoch_flagResponse$trial <- as.numeric(gsub(".*_Trial(\\d+)_.*", "\\1", eye_preproc$epoch_flagResponse$matched_event))
    eye_preproc$epoch_flagTrialstart$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagTrialstart$trial]
    eye_preproc$epoch_flagResponse$baseline <- eye_preproc$baseline_pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly_sub_bl_corr_epoch_flagTrialstart$baseline_means_by_epoch[eye_preproc$epoch_flagResponse$trial]
    eye_preproc$epoch_flagTrialstart$ps_preprocessed <- (eye_preproc$epoch_flagTrialstart$pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly - eye_preproc$epoch_flagTrialstart$baseline) / (eye_preproc$epoch_flagTrialstart$baseline + mean_pupil_size)
    eye_preproc$epoch_flagResponse$ps_preprocessed <- (eye_preproc$epoch_flagResponse$pupil_raw_deblink_detransient_interpolate_lpfilt_detrend_poly - eye_preproc$epoch_flagResponse$baseline) / (eye_preproc$epoch_flagResponse$baseline + mean_pupil_size)
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
    blink_info <- create_blink_info(eye_preproc, subject_num, run_num)
    blink_output_filename <- paste0(subject_id, "_", run_id, "_blink_info.csv")
    write.csv(blink_info, file.path(output_dir, blink_output_filename), row.names = FALSE)
}