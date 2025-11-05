#' Single-response partial least squares regression (PLS1)
#'
#' These helpers expose optimised dense and streaming solvers tailored for
#' partial least squares regression problems where the response consists of a
#' single column. They wrap the high performance C++ routines shipped with the
#' package and provide a user friendly entry point when benchmarking the
#' available implementations.
#' 
#' @name pls1_dense
#' @aliases pls1_dense pls1_stream
#'
#' @param X A `bigmemory::big.matrix` storing the design matrix.
#' @param y Numeric vector of responses with length `nrow(X)`.
#' @param ncomp Number of latent components to compute.
#' @param center Should the columns of `X` be centered? Defaults to `TRUE`.
#' @param scale Should the columns of `X` be scaled to unit variance? Defaults to `FALSE`.
#' @param center_y Should the response be centered? Defaults to `TRUE`.
#' @param scale_y Should the response be scaled to unit variance? Defaults to `FALSE`.
#' @param algorithm Algorithm to use for the fit. Either "simpls" or "nipals".
#'   When choosing "simpls", preprocessing options must remain at their default
#'   values.
#' @param chunk_size Number of rows to process per chunk. Must be strictly
#'   positive. Smaller chunks reduce peak memory usage while larger chunks may
#'   improve speed.
#' @param return_big Logical; when `TRUE`, the coefficients, scores and loadings
#'   are returned as [`bigmemory::big.matrix`] objects. Defaults to `FALSE`.
#'
#' @return A list containing regression coefficients, intercept, latent scores,
#'   loadings and weights.
#'   
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- matrix(rnorm(100), ncol = 1)
#' fit <- pls1_dense(X, y, ncomp = 3)
#' str(fit)
#' }
#' 
# Internal helper to enforce a consistent structure across PLS backends
.harmonize_pls_result <- function(res) {
  if (!is.list(res)) {
    return(res)
  }
  
  if (!is.null(res$coefficients)) {
    coef <- res$coefficients
    coef_vec <- as.numeric(coef)
    if (length(coef_vec) == 0L) {
      res$coefficients <- matrix(numeric(0), nrow = 0L, ncol = 1L)
    } else {
      n_row <- if (is.matrix(coef)) nrow(coef) else length(coef_vec)
      res$coefficients <- matrix(coef_vec, nrow = n_row, ncol = 1L)
    }
  }
  
  if (!is.null(res$y_loadings)) {
    y_load_vec <- as.numeric(res$y_loadings)
    if (length(y_load_vec) == 0L) {
      res$y_loadings <- matrix(numeric(0), nrow = 0L, ncol = 1L)
    } else {
      res$y_loadings <- matrix(y_load_vec, nrow = length(y_load_vec), ncol = 1L)
    }
  }
  if (!"y_loadings" %in% names(res)) {
    res["y_loadings"] <- list(NULL)
  }
  
  if (!is.null(res$weights) && is.null(res$x_weights)) {
    res$x_weights <- res$weights
  }
  if (!is.null(res$loadings) && is.null(res$x_loadings)) {
    res$x_loadings <- res$loadings
  }
  
  if (!is.null(res$x_weights)) {
    res$x_weights <- as.matrix(res$x_weights)
    res$weights <- res$x_weights
  } else if (!is.null(res$weights)) {
    res$weights <- as.matrix(res$weights)
    res$x_weights <- res$weights
  }
  
  if (!is.null(res$x_loadings)) {
    res$x_loadings <- as.matrix(res$x_loadings)
    res$loadings <- res$x_loadings
  } else if (!is.null(res$loadings)) {
    res$loadings <- as.matrix(res$loadings)
    res$x_loadings <- res$loadings
  }
  
  if (!is.null(res$x_means)) {
    res$x_means <- as.numeric(res$x_means)
    res$x_center <- res$x_means
  } else if (!is.null(res$x_center)) {
    res$x_center <- as.numeric(res$x_center)
    res$x_means <- res$x_center
  }
  
  if (!is.null(res$y_mean)) {
    res$y_mean <- as.numeric(res$y_mean)
    res$y_center <- res$y_mean
  } else if (!is.null(res$y_center)) {
    res$y_center <- as.numeric(res$y_center)
    res$y_mean <- res$y_center
  }
  
  if (!is.null(res$x_scales)) {
    res$x_scale <- res$x_scales
    res$x_scales <- NULL
  }
  if (!is.null(res$x_scale)) {
    x_scale <- as.numeric(res$x_scale)
    if (length(x_scale) == 0L || all(x_scale == 1)) {
      res$x_scale <- NULL
    } else {
      res$x_scale <- x_scale
    }
  }
  if (!"x_scale" %in% names(res) || is.null(res$x_scale)) {
    res["x_scale"] <- list(res$x_scale)
  }
  
  if (!is.null(res$y_scales)) {
    res$y_scale <- res$y_scales
    res$y_scales <- NULL
  }
  if (!is.null(res$y_scale)) {
    y_scale <- as.numeric(res$y_scale)
    if (length(y_scale) == 0L || all(y_scale == 1)) {
      res$y_scale <- NULL
    } else {
      res$y_scale <- y_scale
    }
  }
  if (!"y_scale" %in% names(res) || is.null(res$y_scale)) {
    res["y_scale"] <- list(res$y_scale)
  }
  
  if (!is.null(res$scores)) {
    res$scores <- as.matrix(res$scores)
  }
  if (!"scores" %in% names(res)) {
    res["scores"] <- list(NULL)
  }
  
  res
}

#' @export
pls1_dense <- function(X, y, ncomp = 2L, center = TRUE, scale = FALSE,
                       center_y = TRUE, scale_y = FALSE,
                       algorithm = c("simpls", "nipals"),
                       return_big = FALSE) {
  algorithm <- match.arg(algorithm)
  if (!inherits(X, "big.matrix")) {
    stop("X must be a big.matrix")
  }
  y <- as.numeric(y)
  ncomp <- as.integer(ncomp)
  if (length(y) != nrow(X)) {
    stop("Response length does not match the number of rows in X")
  }
  res <- if (identical(algorithm, "simpls")) {
    if (!center || scale || !center_y || scale_y) {
      stop("SIMPLS backend only supports the default centering/scaling options")
    }
    y_mat <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
    y_mat[, 1] <- y
    cpp_big_pls_fit(X@address, y_mat@address, ncomp, 1e-8, return_big)
  } else {
    big_pls_fit_cpp(X@address, y, ncomp, center, scale, center_y, scale_y, return_big)
  }
  .harmonize_pls_result(res)
}

#' @rdname pls1_dense
#'
#' @export
#' 
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- matrix(rnorm(100), ncol = 1)
#' fit <- pls1_stream(X, y, ncomp = 3)
#' str(fit)
#' }
#' 
pls1_stream <- function(X, y, ncomp = 2L, chunk_size = 1024L,
                               center = TRUE, scale = FALSE,
                        center_y = TRUE, scale_y = FALSE,
                        algorithm = c("simpls", "nipals"),
                        return_big = FALSE) {
  algorithm <- match.arg(algorithm)
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
  res <- if (identical(algorithm, "simpls")) {
    if (!center || scale || !center_y || scale_y) {
      stop("SIMPLS backend only supports the default centering/scaling options")
    }
    y_mat <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
    y_mat[, 1] <- y
    cpp_big_pls_stream_fit(X@address, y_mat@address, ncomp, chunk_size, 1e-8, return_big)
  } else {
    big_pls_stream_cpp(X@address, y, ncomp, center, scale, center_y, scale_y, chunk_size, return_big)
  }
  .harmonize_pls_result(res)
}








#' Single-response partial least squares regression (PLS1) another implementation
#'
#' These helpers wrap high-performance C++ routines built on top of the
#' `bigmemory` and `bigalgebra` infrastructure. The `pls1_dense_ya`
#' function performs a standard PLS regression using a NIPALS-style algorithm
#' without copying the data in memory. The `pls1_stream_ya` variant iterates
#' over the data in blocks which makes it possible to handle out-of-core
#' datasets efficiently.
#'
#' @aliases pls1_dense_a pls1_stream_a
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
#' @param chunk_size Number of rows processed at a time by the streaming
#'   backend.
#' @param algorithm Algorithm used to compute the PLS fit. Either "simpls" or
#'   "nipals". The SIMPLS backend only supports the default centering and
#'   scaling configuration.
#' @param return_big Logical; when `TRUE`, the coefficients, scores and loadings
#'   are returned as [`bigmemory::big.matrix`] objects. Defaults to `FALSE`.
#'
#' @return A list with regression coefficients, intercept, latent scores,
#'   weights and additional metadata.
#' 
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
#' fit <- pls1_dense_a(X, y, ncomp = 3)
#' str(fit)
#' }
#' 
#' @export
pls1_dense_a <- function(X, y, ncomp = 2L, center = TRUE, scale = FALSE,
                         tol = 1e-8, max_iter = 100L,
                         algorithm = c("simpls", "nipals"),
                         return_big = FALSE) {
  algorithm <- match.arg(algorithm)
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
  res <- if (identical(algorithm, "simpls")) {
    if (!center || scale) {
      stop("SIMPLS backend only supports center = TRUE and scale = FALSE")
    }
    cpp_big_pls_fit(X@address, y@address, as.integer(ncomp), to, return_bigl)
  } else {
    pls_nipals_bigmemory(X@address, y@address,
                         as.integer(ncomp), center, scale, tol,
                         as.integer(max_iter), return_big)
  }
  res <- .harmonize_pls_result(res)
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}

#' @rdname pls1_dense_a
#' @export
#' 
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
#' fit <- pls1_stream_a(X, y, ncomp = 3)
#' str(fit)
#' }
#' 
pls1_stream_a <- function(X, y, ncomp = 2L, chunk_size = 1024L,
                          center = TRUE, scale = FALSE, tol = 1e-8,
                          algorithm = c("simpls", "nipals"),
                          return_big = FALSE) {
  algorithm <- match.arg(algorithm)
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
  if (!is.numeric(chunk_size) || length(chunk_size) != 1L || chunk_size <= 0) {
    stop("`chunk_size` must be a positive integer")
  }
  res <- if (identical(algorithm, "simpls")) {
    if (!center || scale) {
      stop("SIMPLS backend only supports center = TRUE and scale = FALSE")
    }
    cpp_big_pls_stream_fit(X@address, y@address, as.integer(ncomp),
                           as.integer(chunk_size), tol, return_big)
  } else {
    pls_streaming_bigmemory(X@address, y@address,
                            as.integer(ncomp), as.integer(chunk_size),
                            center, scale, tol, return_big)
  }
  res <- .harmonize_pls_result(res)
  if(algorithm=="simpls"){
    res$chunk_size <- chunk_size
  }
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}











#' Single-response partial least squares regression (PLS1) yet another implementation
#' 
#' @aliases pls1_dense_ya pls1_stream_ya
#' 
#' @param x,y Predictor and response objects stored as double precision
#'   [`bigmemory::big.matrix`] instances. The response must contain a single
#'   column. The dense helper also accepts numeric vectors for `y` and converts
#'   them transparently. The dense routine copies the predictors into an R
#'   matrix, while the streaming version accesses them in blocks.
#' @param ncomp Number of latent components to extract.
#' @param tol Convergence tolerance used when estimating each component. Only
#'   relevant for the dense variant.
#' @param algorithm Algorithm used to compute the PLS fit. Either "simpls" or
#'   "nipals". The SIMPLS backend is generally faster when the data fits in
#'   memory.
#' @param chunk_size Number of rows processed per block by the streaming
#'   variant.
#' @param return_big Logical; when `TRUE`, the coefficients, scores and loadings
#'   are returned as [`bigmemory::big.matrix`] objects. Defaults to `FALSE`.
#'   
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
#' fit <- pls1_dense_ya(X, y, ncomp = 3)
#' str(fit)
#' }
#'
#' @return A list containing regression coefficients, intercept, loadings and
#'   preprocessing statistics. The structure matches the output of the
#'   underlying C++ routines.
#'   
#' @export
pls1_dense_ya <- function(x, y, ncomp, tol = 1e-8,
                          algorithm = c("simpls", "nipals"),
                          return_big = FALSE) {
  algorithm <- match.arg(algorithm)
  if (!bigmemory::is.big.matrix(x)) {
    stop("x must be a big.matrix")
  }
  original_y <- NULL
  if (!bigmemory::is.big.matrix(y)) {
    if (is.numeric(y)) {
      original_y <- as.numeric(y)
      y <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
      y[, 1] <- original_y
    } else {
      stop("y must be either a big.matrix or a numeric vector")
    }
  }
  if (ncol(y) != 1L) {
    stop("y must have a single column")
  }
  if (ncomp <= 0) {
    stop("ncomp must be positive")
  }
  if (ncol(x) == 0L) {
    stop("x must have at least one column")
  }
  res <- if (identical(algorithm, "simpls")) {
    cpp_big_pls_fit(x@address, y@address, as.integer(ncomp), tol, return_big)
  } else {
    y_vec <- if (!is.null(original_y)) original_y else as.numeric(y[, 1])
    big_pls_fit_cpp(x@address, y_vec, as.integer(ncomp),
                    center_x = TRUE, scale_x = FALSE,
                    center_y = TRUE, scale_y = FALSE,
                    return_big = return_big)
  }
  res <- .harmonize_pls_result(res)
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}

#' @rdname pls1_dense_ya
#' 
#' @export
#' 
#' @examples
#' \donttest{
#' library(bigmemory)
#' X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
#' y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
#' fit <- pls1_stream_ya(X, y, ncomp = 3)
#' str(fit)
#' }
#' 
pls1_stream_ya <- function(x, y, ncomp, chunk_size = 4096, tol = 1e-8,
                           algorithm = c("simpls", "nipals"),
                           return_big = FALSE) {
  algorithm <- match.arg(algorithm)
  if (!bigmemory::is.big.matrix(x)) {
    stop("x must be a big.matrix")
  }
  original_y <- NULL
  if (!bigmemory::is.big.matrix(y)) {
    if (is.numeric(y)) {
      original_y <- as.numeric(y)
      y <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
      y[, 1] <- original_y
    } else {
      stop("y must be either a big.matrix or a numeric vector")
    }
  }
  if (ncol(y) != 1L) {
    stop("y must have a single column")
  }
  if (chunk_size <= 0) {
    stop("chunk_size must be positive")
  }
  if (ncomp <= 0) {
    stop("ncomp must be positive")
  }
  res <- if (identical(algorithm, "simpls")) {
    cpp_big_pls_stream_fit(x@address, y@address, as.integer(ncomp),
                           as.integer(chunk_size), tol, return_big)
  } else {
    y_vec <- if (!is.null(original_y)) original_y else as.numeric(y[, 1])
    big_pls_stream_cpp(x@address, y_vec, as.integer(ncomp),
                       center_x = TRUE, scale_x = FALSE,
                       center_y = TRUE, scale_y = FALSE,
                       chunk_size = as.integer(chunk_size),
                       return_big = return_big)
  }
  res <- .harmonize_pls_result(res)
  res$call <- match.call()
  class(res) <- c("big_plsr", class(res))
  res
}
