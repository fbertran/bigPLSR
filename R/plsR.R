#' Fast partial least squares regression for big.matrix inputs
#'
#' @param x A double-precision [`bigmemory::big.matrix`] containing the
#'   predictors. The data is copied into a dense matrix for maximal speed,
#'   therefore this function should be used when the full dataset fits in RAM.
#' @param y A single-column [`bigmemory::big.matrix`] or a numeric vector with
#'   the response values.
#' @param ncomp Number of latent components to extract. Must be positive and no
#'   larger than `ncol(x)`.
#' @param tol Convergence tolerance used when building the latent components.
#'
#' @return A list containing regression coefficients, intercept, loadings, and
#'   centering information. The `coefficients` entry corresponds to the
#'   regression vector and `intercept` stores the bias term. The list also
#'   contains the component loadings (`x_loadings`), weights (`x_weights`), the
#'   response loadings (`y_loadings`), and the centering statistics used during
#'   the fit.
#' @export
#'
#' @examples
#' \dontrun{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
#' fit <- big_pls_fit(X, y, ncomp = 3)
#' str(fit)
#' }
big_pls_fit <- function(x, y, ncomp, tol = 1e-8) {
  if (!bigmemory::is.big.matrix(x)) {
    stop("x must be a big.matrix")
  }
  if (!bigmemory::is.big.matrix(y)) {
    if (is.numeric(y)) {
      y <- bigmemory::as.big.matrix(matrix(y, ncol = 1))
    } else {
      stop("y must be either a big.matrix or a numeric vector")
    }
  }
  if (ncomp <= 0) {
    stop("ncomp must be positive")
  }
  if (ncol(x) == 0L) {
    stop("x must have at least one column")
  }
  invisible(.Call(`_bigPLSR_cpp_big_pls_fit`, x@address, y@address, as.integer(ncomp), tol))
}

#' Streaming partial least squares regression for massive datasets
#'
#' @param x A double-precision [`bigmemory::big.matrix`] with predictor values.
#'   The matrix can be file-backed; data is processed in sequential chunks.
#' @param y A single-column [`bigmemory::big.matrix`] or numeric vector with the
#'   response variable.
#' @param ncomp Number of latent components to compute.
#' @param chunk_size Number of rows processed per streaming block. Larger values
#'   amortise overhead but increase memory usage.
#' @param tol Convergence tolerance for the latent component construction.
#'
#' @return A list with the same structure as [`big_pls_fit()`], produced without
#'   loading the entire dataset into memory.
#' @export
#'
#' @examples
#' \dontrun{
#' library(bigmemory)
#' X <- filebacked.big.matrix(nrow = 1e5, ncol = 50, type = "double")
#' y <- filebacked.big.matrix(nrow = 1e5, ncol = 1, type = "double")
#' fit <- big_pls_stream(X, y, ncomp = 5, chunk_size = 5000)
#' }
big_pls_stream <- function(x, y, ncomp, chunk_size = 4096, tol = 1e-8) {
  if (!bigmemory::is.big.matrix(x)) {
    stop("x must be a big.matrix")
  }
  if (!bigmemory::is.big.matrix(y)) {
    if (is.numeric(y)) {
      y <- bigmemory::as.big.matrix(matrix(y, ncol = 1))
    } else {
      stop("y must be either a big.matrix or a numeric vector")
    }
  }
  if (chunk_size <= 0) {
    stop("chunk_size must be positive")
  }
  if (ncomp <= 0) {
    stop("ncomp must be positive")
  }
  invisible(.Call(`_bigPLSR_cpp_big_pls_stream_fit`, x@address, y@address,
                  as.integer(ncomp), as.integer(chunk_size), tol))
}








#' Partial least squares regression for big.matrix objects
#'
#' @param X A `big.matrix` (double) containing the predictor variables.
#' @param Y A `big.matrix` (double) containing the response variables.
#' @param ncomp Number of latent components to compute.
#' @param center Should the columns be centered before fitting?
#' @param scale Should the columns be scaled to unit variance before fitting?
#'
#' @return A list with regression coefficients, intercept, weights, loadings and
#'   additional preprocessing statistics.
#' @export
big_plsr <- function(X, Y, ncomp, center = TRUE, scale = FALSE) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  if (!inherits(Y, "big.matrix")) {
    stop("Y must be a big.matrix")
  }
  big_plsr_fit(X@address, Y@address, as.integer(ncomp), center, scale)
}

#' Streaming partial least squares regression for big.matrix objects
#'
#' This variant processes the data in blocks so that datasets that are larger
#' than the available RAM can be analysed.
#'
#' @inheritParams big_plsr
#' @param block_size Number of rows processed at once. Larger values typically
#'   speed up the computation when enough memory is available.
#'
#' @return A list containing the fitted model parameters and preprocessing
#'   statistics. The number of computed components is returned in the `ncomp`
#'   element and may be smaller than the requested number if convergence failed.
#' @export
big_plsr_stream <- function(X, Y, ncomp, center = TRUE, scale = FALSE, block_size = 1024L) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  if (!inherits(Y, "big.matrix")) {
    stop("Y must be a big.matrix")
  }
  big_plsr_stream_fit(X@address, Y@address, as.integer(ncomp), center, scale, as.integer(block_size))
}






#' Fit partial least squares regression on a big.matrix
#'
#' @param X A `bigmemory::big.matrix` storing the design matrix.
#' @param y Numeric vector of responses with length `nrow(X)`.
#' @param ncomp Number of latent components to compute.
#' @param center Should the columns of `X` be centered? Defaults to `TRUE`.
#' @param scale Should the columns of `X` be scaled to unit variance? Defaults to `FALSE`.
#' @param center_y Should the response be centered? Defaults to `TRUE`.
#' @param scale_y Should the response be scaled to unit variance? Defaults to `FALSE`.
#'
#' @return A list containing regression coefficients, intercept, latent scores,
#'   loadings and weights.
#' @export
big_pls_fit_big <- function(X, y, ncomp = 2L, center = TRUE, scale = FALSE,
                        center_y = TRUE, scale_y = FALSE) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  y <- as.numeric(y)
  ncomp <- as.integer(ncomp)
  if (length(y) != nrow(X)) {
    stop("Response length does not match the number of rows in X")
  }
  big_pls_fit_cpp(X@address, y, ncomp, center, scale, center_y, scale_y)
}

#' Stream-friendly partial least squares fit
#'
#' @inheritParams big_pls_fit_big
#' @param chunk_size Number of rows to process per chunk. Must be strictly
#'   positive. Smaller chunks reduce peak memory usage while larger chunks may
#'   improve speed.
#'
#' @return Same as [big_pls_fit()].
#' @export
big_pls_big_stream_fit <- function(X, y, ncomp = 2L, chunk_size = 1024L,
                               center = TRUE, scale = FALSE,
                               center_y = TRUE, scale_y = FALSE) {
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  y <- as.numeric(y)
  ncomp <- as.integer(ncomp)
  chunk_size <- as.integer(chunk_size)
  if (length(y) != nrow(X)) {
    stop("Response length does not match the number of rows in X")
  }
  if (chunk_size <= 0L) {
    stop("chunk_size must be a positive integer")
  }
  big_pls_stream_cpp(X@address, y, ncomp, center, scale, center_y, scale_y, chunk_size)
}

















#' Partial Least Squares regression for big.matrix objects
#'
#' These helpers wrap high-performance C++ routines built on top of the
#' `bigmemory` and `bigalgebra` infrastructure. The `pls_bigmemory_fit`
#' function performs a standard PLS regression using a NIPALS-style algorithm
#' without copying the data in memory. The `pls_streaming_fit` variant iterates
#' over the data in blocks which makes it possible to handle out-of-core
#' datasets efficiently.
#'
#' @param X A `big.matrix` object containing the predictors.
#' @param y Either a `big.matrix` with a single column or a numeric vector with
#'   the response values.
#' @param ncomp Number of latent components to extract.
#' @param center Logical; should the predictors and response be centered.
#' @param scale Logical; should the predictors and response be scaled to unit
#'   variance before fitting the model.
#' @param tol Numerical tolerance used to detect convergence breakdown.
#' @param max_iter Maximum number of iterations for the internal solver (kept
#'   for compatibility; the solver adapts automatically when convergence issues
#'   are detected).
#' @param block_size Number of rows processed at a time by the streaming
#'   backend.
#'
#' @return A list with regression coefficients, intercept, latent scores,
#'   weights and additional metadata.
#' @export
pls_bigmemory_fit <- function(X, y, ncomp = 2L, center = TRUE, scale = FALSE,
                              tol = 1e-8, max_iter = 100L) {
  if (!inherits(X, "big.matrix")) {
    stop("`X` must be a big.matrix")
  }
  if (!inherits(y, "big.matrix")) {
    if (is.numeric(y)) {
      y_mat <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
      y_mat[, 1] <- y
      y <- y_mat
    } else {
      stop("`y` must be a big.matrix or a numeric vector")
    }
  }
  if (nrow(X) != nrow(y)) {
    stop("`X` and `y` must have matching numbers of rows")
  }
  if (ncol(y) != 1L) {
    stop("`y` must have a single column")
  }
  res <- pls_nipals_bigmemory(X@address, y@address,
                              as.integer(ncomp), center, scale, tol,
                              as.integer(max_iter))
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}

#' @rdname pls_bigmemory_fit
#' @export
pls_streaming_fit <- function(X, y, ncomp = 2L, block_size = 1024L,
                              center = TRUE, scale = FALSE, tol = 1e-8) {
  if (!inherits(X, "big.matrix")) {
    stop("`X` must be a big.matrix")
  }
  if (!inherits(y, "big.matrix")) {
    if (is.numeric(y)) {
      y_mat <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
      y_mat[, 1] <- y
      y <- y_mat
    } else {
      stop("`y` must be a big.matrix or a numeric vector")
    }
  }
  if (nrow(X) != nrow(y)) {
    stop("`X` and `y` must have matching numbers of rows")
  }
  if (ncol(y) != 1L) {
    stop("`y` must have a single column")
  }
  if (!is.numeric(block_size) || length(block_size) != 1L || block_size <= 0) {
    stop("`block_size` must be a positive integer")
  }
  res <- pls_streaming_bigmemory(X@address, y@address,
                                 as.integer(ncomp), as.integer(block_size),
                                 center, scale, tol)
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}
