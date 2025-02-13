library(mgcv)

# Function to remove the gaze position regressor from the pupil data
remove_gaze_regressor <- function(data) {
  # Step 1: Run the model (bam) to regress out the eye position
  m1 <- bam(
    pupil_raw ~ s(eye_x, eye_y, bs = "tp", k = 25),
    data = data$timeseries,
    method = "fREML",
    discrete = TRUE,
    na.action = na.exclude
  )

  # Step 2: Get residuals
  residuals_ps <- resid(m1)

  # Step 3: Add the mean of the original pupil size to the residuals
  data$timeseries$pupil_raw_orig <- data$timeseries$pupil_raw
  mean_pupil_raw <- mean(data$timeseries$pupil_raw_orig, na.rm = TRUE)
  data$timeseries$pupil_raw <- residuals_ps + mean_pupil_raw

  return(data)
}