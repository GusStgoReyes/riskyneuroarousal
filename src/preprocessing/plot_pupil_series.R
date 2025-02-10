library(ggplot2)

# Plot and save the pupil time series
plot_pupil_series <- function(data,
                              output_dir = ".",
                              filename = "pupil_series_plot.png") {
  # Convert time to seconds
  time_orig <- data$timeseries$time_orig
  start_time <- min(time_orig)
  data$timeseries$time <- (time_orig - start_time) / 1000
  p <- ggplot(data$timeseries, aes(x = time, y = pupil_raw_deblink_detransient_interpolate_lpfilt)) +
    geom_line(color = "blue") +
    labs(
      x = "Time (seconds)",
      y = "A.U. Change from Mean"
    ) +
    theme_minimal()

  # Save the plot
  ggsave(filename, plot = p, width = 12, height = 4, units = "in", path = output_dir)
}