# Replace any pupil_size values that are out of bounds to NaN
replace_out_of_bounds <- function(data,
                                  screen_width = 1920,
                                  screen_height = 1080,
                                  focus_screen_width = 1000,
                                  focus_screen_height = 500) {
  data$timeseries$pupil_raw[data$timeseries$eye_x < screen_width/2 - focus_screen_width/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_x > screen_width/2 + focus_screen_width/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_y < screen_height/2 - focus_screen_height/2] <- NA
  data$timeseries$pupil_raw[data$timeseries$eye_y > screen_height/2 + focus_screen_height/2] <- NA

  return(data)
}