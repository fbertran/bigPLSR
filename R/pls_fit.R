
#' Unified PLS fit with auto backend
#'
#' Dispatches to a dense (Arm/BLAS) backend for in-memory matrices
#' or to a streaming big.matrix backend when X (or Y) is a big.matrix.
#'
#' @param X numeric matrix or \code{bigmemory::big.matrix}
#' @param y numeric vector or single-column \code{big.matrix}
#' @param ncomp number of latent components
#' @param tol numeric tolerance used in the core solver
#' @param backend one of \code{"auto"}, \code{"arma"}, \code{"bigmem"}
#' @param scores one of \code{"none"}, \code{"r"}, \code{"big"}
#' @param chunk_size chunk size for the bigmem backend
#' @param scores_name name for the scores big.matrix (when \code{scores="big"})
#' @return a list with coefficients, intercept, weights, loadings, means,
#'   and optionally \code{$scores}.
#' @export
pls_fit <- function(X, y, ncomp, tol = 1e-8,
                    backend = c("auto","arma","bigmem"),
                    scores = c("none","r","big"),
                    chunk_size = 10000L,
                    scores_name = "scores") {
  backend <- match.arg(backend)
  scores  <- match.arg(scores)

  is_big <- inherits(X, "big.matrix") || inherits(X, "big.matrix.descriptor")
  if (backend == "auto") {
    if (is_big) backend <- "bigmem" else backend <- "arma"
  }

  if (backend == "arma") {
    if (is_big) {
      # copy from big.matrix to R matrix for speed on small data
      Xr <- as.matrix(X[])
      yr <- if (inherits(y, "big.matrix")) as.numeric(y[,1]) else as.numeric(y)
    } else {
      Xr <- as.matrix(X)
      yr <- as.numeric(y)
    }
    compute_scores <- scores != "none"
    scores_big <- identical(scores, "big")
    return(.Call(`_bigPLSR_cpp_dense_pls_fit`, Xr, yr, as.integer(ncomp), tol,
                 compute_scores, scores_big, as.character(scores_name)))
  } else {
    # bigmem backend; use existing streaming function
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix")) stop("For backend='bigmem', y must be a single-column big.matrix")
    if (ncol(y) != 1) stop("y big.matrix must have one column")
    if (!identical(scores, "none")) {
      # Future: implement streaming scores
      warning("scores for bigmem backend not implemented yet; returning model only")
    }
    return(.Call(`_bigPLSR_cpp_big_pls_stream_fit`, X@address, y@address,
                 as.integer(ncomp), as.integer(chunk_size), tol, FALSE))
  }
}
