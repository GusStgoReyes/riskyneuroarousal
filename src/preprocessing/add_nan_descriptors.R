add_nan_descriptors <- function(data,
                                screen_width = 1920,
                                screen_height = 1080,
                                focus_screen_width = 1000,
                                focus_screen_height = 500) {

  # Add indicator variable for blink
  data$timeseries$blink <- is.na(data$timeseries$pupil_raw)
  # Add indicator variable for out of bounds
  data$timeseries$outofbounds <-
    (data$timeseries$eye_x < screen_width / 2 - focus_screen_width / 2 |
     data$timeseries$eye_x > screen_width / 2 + focus_screen_width / 2 |
     data$timeseries$eye_y < screen_height / 2 - focus_screen_height / 2 |
     data$timeseries$eye_y > screen_height / 2 + focus_screen_height / 2)
  return(data)
}