`%||%` <- function(a,b) if (is.null(a)) b else a 

maybe_convert_to_bigmatrix <- function(result, components, return_big) {
  if (!isTRUE(return_big) || !is.list(result)) {
    return(result)
  }
  for (component in components) {
    value <- result[[component]]
    if (is.null(value) || inherits(value, "big.matrix")) {
      next
    }
    result[[component]] <- as_bigmatrix(value)
  }
  result
}

as_bigmatrix <- function(object) {
  if (inherits(object, "big.matrix") || is.null(object)) {
    return(object)
  }
  if (is.data.frame(object)) {
    object <- as.matrix(object)
  }
  dims <- dim(object)
  if (is.null(dims)) {
    vec <- as.numeric(object)
    if (!length(vec)) {
      return(object)
    }
    bm <- bigmemory::big.matrix(nrow = length(vec), ncol = 1L, type = "double")
    bm[, 1] <- vec
    return(bm)
  }
  if (length(dims) != 2L || any(dims == 0L)) {
    return(object)
  }
  mat <- as.matrix(object)
  bm <- bigmemory::big.matrix(nrow = dims[1], ncol = dims[2], type = "double")
  if (length(mat)) {
    bm[,] <- mat
  }
  bm
}
