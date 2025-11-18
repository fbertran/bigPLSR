.onLoad <- function(...) {
  op <- options()
  defaults <- list(
    bigPLSR.simpls.cross = "auto",
    bigPLSR.simpls.solve = "chol"  # choose "solve" if you want conservative fallback
  )
  toset <- !(names(defaults) %in% names(op))
  if (any(toset)) options(defaults[toset])
}
