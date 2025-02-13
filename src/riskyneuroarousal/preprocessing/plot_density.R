library(ggplot2)

# Function to create and save a density plot of the time spent at locations
plot_density <- function(data,
                         output_dir = ".",
                         filename = "density_plot.png") {
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