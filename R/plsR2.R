#' Partial least squares regression for multi-response problems (PLS2)
#'
#' @param X A `bigmemory::big.matrix` containing the predictor variables.
#' @param Y A `bigmemory::big.matrix` storing the multi-dimensional response.
#' @param ncomp Number of latent components to compute.
#' @param center Should the inputs be centered prior to fitting?
#' @param scale Should the inputs be scaled to unit variance prior to fitting?
#' @param algorithm PLS backend to use. Either "simpls" (default) or "nipals".
#' @param chunk_size Number of rows processed per block by the streaming
#'   variant.
#' @param return_big Logical; when `TRUE`, the coefficients, scores and loadings
#'   are returned as [`bigmemory::big.matrix`] objects. Defaults to `FALSE`.
#'
#' @return A list with regression coefficients, intercept, weights, loadings
#'   and preprocessing metadata.
#'
#' @export
pls2_dense <- function(X, Y, ncomp, center = TRUE, scale = FALSE,
                       algorithm = c("simpls", "nipals"),
                       return_big = FALSE) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  if (!inherits(Y, "big.matrix")) {
    stop("Y must be a big.matrix")
  }
  algorithm <- match.arg(algorithm)
  res <- if (algorithm == "simpls") {
    big_plsr_fit(X@address, Y@address, as.integer(ncomp), center, scale, return_big)
  } else {
    big_plsr_fit_nipals(X@address, Y@address, as.integer(ncomp), center, scale, return_big)
  }
  res
}

#' @rdname pls2_dense
#' 
#' @export
pls2_stream <- function(X, Y, ncomp, center = TRUE, scale = FALSE,
                        chunk_size = 1024L,
                        algorithm = c("simpls", "nipals"),
                        return_big = FALSE) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  if (!inherits(Y, "big.matrix")) {
    stop("Y must be a big.matrix")
  }
  algorithm <- match.arg(algorithm)
  res <- if (algorithm == "simpls") {
    big_plsr_stream_fit(X@address, Y@address, as.integer(ncomp), center, scale, as.integer(chunk_size), return_big)
  } else {
    big_plsr_stream_fit_nipals(X@address, Y@address, as.integer(ncomp), center, scale, as.integer(chunk_size), return_big)
  }
  res
}


