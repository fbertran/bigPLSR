
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
#' 
#' @examples
#' bmX <- bigmemory::as.big.matrix(X)
#' bmy <- bigmemory::as.big.matrix(matrix(y, nrow(X), 1))
#' sink_bm <- bigmemory::filebacked.big.matrix(
#'   nrow = nrow(bmX), ncol = 3, type = "double",
#'   backingfile = "scores.bin", backingpath = tempdir(),
#'   descriptorfile = "scores.desc"
#' )
#'
#' fit <- pls_fit(bmX, bmy, ncomp = 3, backend = "bigmem", scores = "big",
#'               scores_target = "existing", scores_bm = sink_bm)
#' 
#' 
pls_fit <- function(X, y, ncomp, tol = 1e-8,
                    backend = c("auto","arma","bigmem"),
                    scores = c("none","r","big"),
                    chunk_size = 10000L,
                    scores_name = "scores") {
  backend <- match.arg(backend)
  scores  <- match.arg(scores)
  scores_target <- match.arg(scores_target)

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
    sink_bm <- NULL
    if (identical(scores, "big")) {
      if (scores_target == "existing") {
        if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix")) {
          sink_bm <- scores_bm
        } else if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix.descriptor")) {
          sink_bm <- bigmemory::attach.big.matrix(scores_bm)
        } else if (!is.null(scores_backingfile)) {
          sink_bm <- bigmemory::filebacked.big.matrix(
            nrow = nrow(X), ncol = as.integer(ncomp), type = "double",
            backingfile = scores_backingfile,
            backingpath = if (is.null(scores_backingpath)) getwd() else scores_backingpath,
            descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile)
        } else {
          stop("scores_target='existing' requires scores_bm or backingfile/path/descriptorfile")
        }
      }
    }
    return(.Call(`_bigPLSR_cpp_big_pls_stream_fit_sink`, X@address, y@address,
                 if (is.null(sink_bm)) NULL else sink_bm,
                 as.integer(ncomp), as.integer(chunk_size), tol, identical(scores,"big")))
  }
}
