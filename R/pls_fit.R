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
#' set.seed(1)
#' X <- matrix(rnorm(200*50), 200, 50)
#' y <- X[,1]*2 - X[,2] + rnorm(200)
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
#' Dense, named scores (no descriptor here; scores are base matrix)
#' fit1 <- pls_fit(X, y, ncomp = 3,
#'                 backend = "arma", scores = "r",
#'                 scores_colnames = c("t1","t2","t3"),
#'                 return_scores_descriptor = TRUE)
#' colnames(fit1$scores)              # "t1" "t2" "t3"
#' "scores_descriptor" %in% names(fit1)  # FALSE (dense scores aren't big.matrix)
#' 
#' # Big-matrix with external file-backed sink, named + descriptor
#' sink_bm <- bigmemory::filebacked.big.matrix(
#'   nrow=nrow(bmX), ncol=3, type="double",
#'   backingfile="scores.bin", backingpath=tempdir(),
#'   descriptorfile="scores.desc"
#' )
#' fit2 <- pls_fit(bmX, bmy, ncomp = 3,
#'                 backend = "bigmem", scores = "big",
#'                 scores_target = "existing", scores_bm = sink_bm,
#'                 scores_colnames = c("t1","t2","t3"),
#'                 return_scores_descriptor = TRUE)
#' colnames(fit2$scores)               # "t1" "t2" "t3"
#' fit2$scores_descriptor              # big.matrix.descriptor
#' 
#' 
pls_fit <- function(
    X, y, ncomp,
    tol = 1e-8,
    backend = c("auto", "arma", "bigmem"),
    scores  = c("none", "r", "big"),
    chunk_size = 10000L,
    scores_name = "scores",
    mode = c("auto","pls1","pls2"),
    scores_target = c("auto","new","existing"),
    scores_bm = NULL,
    scores_backingfile = NULL,
    scores_backingpath = NULL,
    scores_descriptorfile = NULL,
    scores_colnames = NULL,
    return_scores_descriptor = FALSE
) {
  backend <- match.arg(backend)
  scores  <- match.arg(scores)
  mode    <- match.arg(mode)
  scores_target <- match.arg(scores_target)
  
  is_big <- inherits(X, "big.matrix") || inherits(X, "big.matrix.descriptor")
  # Auto mode based on y dimension
  y_ncol <- if (is.matrix(y)) ncol(y) else if (inherits(y, "big.matrix")) ncol(y) else 1L
  if (mode == "auto") mode <- if (y_ncol == 1L) "pls1" else "pls2"
  if (backend == "auto") {
    if (is_big) backend <- "bigmem" else backend <- "arma"
  }
  
  if (backend == "arma") {
    if (is_big) {
      # copy from big.matrix to R matrix for speed on small data
      Xr <- as.matrix(X[])
      yr <- if (inherits(y, "big.matrix")) {
        if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[,1])
      } else {
        as.numeric(y)
      }
    } else {
      Xr <- as.matrix(X)
      yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) y else as.numeric(y)
    }
    ## ---- DENSE ROUTING: pls1 (fast) vs pls2 (SIMPLS on cross-products) ----
    if (mode == "pls1" || (!is.matrix(yr))) {
      compute_scores <- scores != "none"
      scores_big <- identical(scores, "big")
      fit <- .Call(`_bigPLSR_cpp_dense_pls_fit`, Xr, yr, as.integer(ncomp), tol,
                   compute_scores, scores_big, as.character(scores_name))
    } else {
      ## PLS2 SIMPLS: use the same cross-product path as bigmem for parity
      Y <- as.matrix(yr)
      ## means
      x_means <- colMeans(Xr)
      y_means <- colMeans(Y)
      ## centered cross-products (dense)
      Xc <- sweep(Xr, 2L, x_means, FUN = "-")
      Yc <- sweep(Y,  2L, y_means, FUN = "-")
      XtX <- crossprod(Xc)         # p x p
      XtY <- crossprod(Xc, Yc)     # p x m
      fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                   XtX, XtY, x_means, y_means, as.integer(ncomp), tol)
      ## Dense scores if requested: T = Xc %*% W
      if (scores != "none") {
        Tmat <- Xc %*% fit$x_weights
        if (identical(scores, "r")) {
          fit$scores <- Tmat
        } else {
          ## scores == "big": allocate/copy to a big.matrix sink if provided
          if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix")) {
            scores_bm[,] <- Tmat
            fit$scores <- scores_bm
          } else if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix.descriptor")) {
            bm <- bigmemory::attach.big.matrix(scores_bm)
            bm[,] <- Tmat
            fit$scores <- bm
          } else if (!is.null(scores_backingfile)) {
            bm <- bigmemory::filebacked.big.matrix(
              nrow = nrow(Tmat), ncol = ncol(Tmat), type = "double",
              backingfile = scores_backingfile,
              backingpath = if (is.null(scores_backingpath)) getwd() else scores_backingpath,
              descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile
            )
            bm[,] <- Tmat
            fit$scores <- bm
          } else {
            ## fall back to in-memory if no sink specified
            fit$scores <- Tmat
          }
        }
      } else {
        fit$scores <- NULL
      }
      fit$mode <- "pls2"
    }
    ## Post-process names/descriptor (dense scores are base matrix -> no descriptor)
    if (!is.null(fit$scores)) {
      used_comp <- fit$ncomp
      if (!is.null(scores_colnames)) {
        nm <- as.character(scores_colnames)
        if (length(nm) != used_comp) {
          warning("length(scores_colnames) != ncomp used; truncating/recycling")
          length(nm) <- used_comp
        }
        colnames(fit$scores) <- nm
      }
      if (isTRUE(return_scores_descriptor) && inherits(fit$scores, "big.matrix")) {
        fit$scores_descriptor <- bigmemory::describe(fit$scores)
      }
    }
    return(fit)  
  } else {
    # bigmem backend with streaming + optional external sink
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix")) stop("For backend='bigmem', y must be a big.matrix")
    if (mode == "pls1" && ncol(y) != 1L) stop("mode='pls1' requires y to have one column")
    
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
            descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile
          )
        } else {
          stop("scores_target='existing' requires scores_bm or backingfile/path/descriptorfile")
        }
      }
    }
    
    ## ---- BIGMEM ROUTING ----
    if (mode == "pls1") {
      fit <- .Call(`_bigPLSR_cpp_big_pls_stream_fit_sink`, X@address, y@address,
                   if (is.null(sink_bm)) NULL else sink_bm,
                   as.integer(ncomp), as.integer(chunk_size), tol, identical(scores, "big"))
    } else {
      ## NEW: PLS2 fast path via cross-products + SIMPLS
      cross <- .Call(`_bigPLSR_cpp_bigmem_cross`, X@address, y@address, as.integer(chunk_size))
      fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                   cross$XtX, cross$XtY, cross$x_means, cross$y_means,
                   as.integer(ncomp), tol)
      
      ## Stream scores if requested (chunked T = (X - mu) %*% W)
      if (identical(scores, "big") || identical(scores, "r")) {
        local_sink <- NULL
        if (identical(scores, "big")) {
          if (is.null(sink_bm)) {
            sink_bm <- bigmemory::filebacked.big.matrix(
              nrow = nrow(X), ncol = as.integer(fit$ncomp), type = "double",
              backingfile = if (is.null(scores_backingfile)) "scores.bin" else scores_backingfile,
              backingpath = if (is.null(scores_backingpath)) getwd() else scores_backingpath,
              descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile
            )
          }
          local_sink <- sink_bm
        }
        fit$scores <- .Call(`_bigPLSR_cpp_stream_scores_given_W`,
                            X@address, fit$x_weights, fit$x_means,
                            as.integer(chunk_size),
                            if (is.null(local_sink)) NULL else local_sink,
                            identical(scores, "big"))
      } else {
        fit$scores <- NULL
      }
    }
    
    ## Post-process names/descriptor for bigmem
    if (!is.null(fit$scores)) {
      used_comp <- fit$ncomp
      if (!is.null(scores_colnames)) {
        nm <- as.character(scores_colnames)
        if (length(nm) != used_comp) {
          warning("length(scores_colnames) != ncomp used; truncating/recycling")
          length(nm) <- used_comp
        }
        colnames(fit$scores) <- nm
      }
      if (isTRUE(return_scores_descriptor) && inherits(fit$scores, "big.matrix")) {
        fit$scores_descriptor <- bigmemory::describe(fit$scores)
      }
    }
    return(fit)
  }
}
