library(stringr)
library(dplyr)
library(eyelinker)
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

for (file in asc_files[9:12]) {
  # Load the data
  subject_id <- stringr::str_extract(file, "sub-\\d{3}")
  run_id <- stringr::str_extract(file, "run-\\d")
  if (!(subject_id %in% valid_subs)) {
    cat("Skipping subject ", subject_id, " as it is not in the valid list.\n")
    next
  }

  loaded_data <- read.asc(file)

  # Valid saccade end times
  sacc_events <- loaded_data$sacc$etime[loaded_data$sacc$exp > 1920/2 - 500 & loaded_data$sacc$exp < 1920/2 + 500 & loaded_data$sacc$eyp > 1080/2 - 250 & loaded_data$sacc$eyp < 1080/2 + 250]

  # Saccade blinks
  blink_events <- loaded_data$blinks$etime + 100

  # Start times
  trial_starts <- loaded_data$msg %>%
    filter(str_detect(text, "flag_TrialStart")) %>%
    mutate(
        trial = str_extract(text, "Trial\\d+") %>% str_remove("Trial") %>% as.integer(),
        start_time = time
    ) %>%
    select(block, trial, start_time)

  # Response
  responses <- loaded_data$msg %>%
    filter(str_detect(text, "flag_Response")) %>%
    mutate(
        trial = str_extract(text, "Trial\\d+") %>% str_remove("Trial") %>% as.integer(),
        response_time = time
    ) %>%
    select(block, trial, response_time)

  # Durations
  trial_durations <- trial_starts %>%
    inner_join(responses, by = c("block", "trial")) %>%
    mutate(duration = response_time - start_time)

  # Save each event type to a separate file csv (using run_id in filename)
  write.csv(sacc_events, file = paste0(output_dir, "/", subject_id, "_", run_id, "_saccades.csv"), row.names = FALSE)
  write.csv(blink_events, file = paste0(output_dir, "/", subject_id, "_", run_id, "_blinks.csv"), row.names = FALSE)
  write.csv(trial_starts, file = paste0(output_dir, "/", subject_id, "_", run_id, "_trial_starts.csv"), row.names = FALSE)
  write.csv(responses, file = paste0(output_dir, "/", subject_id, "_", run_id, "_responses.csv"), row.names = FALSE)
  write.csv(trial_durations, file = paste0(output_dir, "/", subject_id, "_", run_id, "_trial_durations.csv"), row.names = FALSE)

}