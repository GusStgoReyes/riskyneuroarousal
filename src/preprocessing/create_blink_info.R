library(dplyr)

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

  trial_blink_summary$trial <- trial_blink_summary$trial + (run_num - 1) * 64
  trial_blink_summary$sub <- sub_num

  return(trial_blink_summary)
}