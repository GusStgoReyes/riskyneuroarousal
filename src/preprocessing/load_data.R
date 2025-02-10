library(eyeris)

load_data <- function(file, valid_subs) {
  # Extract subject_id and run_id
  subject_id <- stringr::str_extract(file, "sub-\\d{3}")
  run_id <- stringr::str_extract(file, "run-\\d")
  subject_num <- as.numeric(stringr::str_extract(subject_id, "\\d{3}"))
  run_num <- as.numeric(stringr::str_extract(run_id, "\\d"))
  print(paste("Processing", subject_id, run_id))

  # Only do processing for valid subjects
  if (!(subject_id %in% valid_subs)) {
    cat("Skipping subject ", subject_id, " as it is not in the valid list.\n")
    return(NULL)
  }

  # Load the data (this will only contain blinks and pupil size)
  data <- eyeris::load_asc(file)

  return(list(
    data = data,
    subject_id = subject_id,
    run_id = run_id,
    subject_num = subject_num,
    run_num = run_num
  ))
}