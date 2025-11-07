#' Unified PLS fit with auto backend and selectable algorithm
#'
#' Dispatches to a dense (Arm/BLAS) backend for in-memory matrices
#' or to a streaming big.matrix backend when X (or Y) is a big.matrix.
#' Algorithm can be chosen between "simpls" (default) and "nipals".
#'
#' @param X numeric matrix or \code{bigmemory::big.matrix}
#' @param y numeric vector/matrix or \code{big.matrix}
#' @param ncomp number of latent components
#' @param tol numeric tolerance used in the core solver
#' @param backend one of \code{"auto"}, \code{"arma"}, \code{"bigmem"}
#' @param mode one of \code{"auto"}, \code{"pls1"}, \code{"pls2"}
#' @param algorithm one of \code{"auto"}, \code{"simpls"}, \code{"nipals"}
#' @param scores one of \code{"none"}, \code{"r"}, \code{"big"}
#' @param chunk_size chunk size for the bigmem backend
#' @param scores_name name for dense scores (or output big.matrix)
#' @param scores_target one of \code{"auto"}, \code{"new"}, \code{"existing"}
#' @param scores_bm optional existing big.matrix or descriptor for scores
#' @param scores_backingfile/backingpath/descriptorfile file-backed sink args
#' @param scores_colnames optional character vector for score column names
#' @param return_scores_descriptor logical; if TRUE and scores is big.matrix, add \code{$scores_descriptor}
#' @return a list with coefficients, intercept, weights, loadings, means,
#'   and optionally \code{$scores}.
#' @export
pls_fit <- function(
    X, y, ncomp,
    tol = 1e-8,
    backend = c("auto", "arma", "bigmem"),
    mode    = c("auto","pls1","pls2"),
    algorithm = c("auto","simpls","nipals"),
    scores  = c("none", "r", "big"),
    chunk_size = 10000L,
    scores_name = "scores",
    scores_target = c("auto","new","existing"),
    scores_bm = NULL,
    scores_backingfile = NULL,
    scores_backingpath = NULL,
    scores_descriptorfile = NULL,
    scores_colnames = NULL,
    return_scores_descriptor = FALSE
) {
  backend   <- match.arg(backend)
  mode      <- match.arg(mode)
  scores    <- match.arg(scores)
  algo_in   <- match.arg(algorithm)
  scores_target <- match.arg(scores_target)
  
  is_big <- inherits(X, "big.matrix") || inherits(X, "big.matrix.descriptor")
  y_ncol <- if (is.matrix(y)) ncol(y) else if (inherits(y, "big.matrix")) ncol(y) else 1L
  if (mode == "auto")    mode    <- if (y_ncol == 1L) "pls1" else "pls2"
  if (backend == "auto") backend <- if (is_big) "bigmem" else "arma"
  
  # Helper: apply colnames + descriptor on scores
  .post_scores <- function(fit) {
    if (!is.null(fit$scores)) {
      used_comp <- if (!is.null(fit$ncomp)) fit$ncomp else if (!is.null(fit$x_weights)) ncol(fit$x_weights) else ncol(fit$scores)
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
    fit
  }
  
  # ---- DENSE BACKEND --------------------------------------------------------
  run_dense_simpls <- function() {
    if (is_big) {
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
    
    if (mode == "pls1" || (!is.matrix(yr))) {
      compute_scores <- scores != "none"
      scores_big <- identical(scores, "big")  # dense path: we copy if "big"
      fit <- .Call(`_bigPLSR_cpp_dense_pls_fit`, Xr, yr, as.integer(ncomp), tol,
                   compute_scores, scores_big, as.character(scores_name))
      fit$mode <- "pls1"
    } else {
      Y <- as.matrix(yr)
      x_means <- as.numeric(colMeans(Xr))
      y_means <- as.numeric(colMeans(Y))
      Xc <- sweep(Xr, 2L, x_means, FUN = "-")
      Yc <- sweep(Y,  2L, y_means, FUN = "-")
      XtX <- crossprod(Xc)
      XtY <- crossprod(Xc, Yc)
      fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                   XtX, XtY, x_means, y_means, as.integer(ncomp), tol)
      fit$mode <- "pls2"
      
      # Dense scores if requested: T = Xc %*% W
      if (scores != "none") {
        Tmat <- Xc %*% fit$x_weights
        if (identical(scores, "r")) {
          fit$scores <- Tmat
        } else {
          # scores=="big" → use user sink or file-backed
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
            fit$scores <- Tmat
          }
        }
      } else {
        fit$scores <- NULL
      }
    }
    .finalize_pls_fit(.post_scores(fit), "simpls")
  }
  
  run_dense_nipals <- function() {
    if (is_big) {
      Xr <- as.matrix(X[])
      if (inherits(y, "big.matrix")) {
        yr <- if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[,1])
      } else {
        yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
      }
    } else {
      Xr <- as.matrix(X)
      yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
    }
    
    compute_scores <- !identical(scores, "none")
    Yin <- if (mode == "pls2") yr else as.numeric(yr)
    
    fit <- cpp_dense_plsr_nipals(
      Xr, Yin, as.integer(ncomp), tol,
      compute_scores = compute_scores,
      scores_big = FALSE, scores_name = as.character(scores_name)
    )
    fit$mode <- mode
    
    # If user asked for "big" scores, copy to a sink.
    if (identical(scores, "big") && !is.null(fit$scores)) {
      Tmat <- as.matrix(fit$scores)
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
      }
    } else if (identical(scores, "none")) {
      fit$scores <- NULL
    }
    
    fit <- .finalize_pls_fit(.post_scores(fit), "nipals")
    if (isTRUE(return_scores_descriptor) && inherits(fit$scores, "big.matrix")) {
      fit$scores_descriptor <- bigmemory::describe(fit$scores)
    }
    fit
  }
  
  # ---- BIGMEM BACKEND -------------------------------------------------------
  run_bigmem_simpls <- function() {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix")) stop("For backend='bigmem', y must be a big.matrix")
    
    if (mode == "pls1") {
      # existing streaming SIMPLS PLS1 path
      sink_bm <- NULL
      if (identical(scores, "big") && scores_target == "existing") {
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
      
      fit <- .Call(`_bigPLSR_cpp_big_pls_stream_fit_sink`, X@address, y@address,
                   if (is.null(sink_bm)) NULL else sink_bm,
                   as.integer(ncomp), as.integer(chunk_size), tol, identical(scores, "big"))
      fit$mode <- "pls1"
    } else {
      # PLS2: chunked cross-products + SIMPLS
      cross <- .Call(`_bigPLSR_cpp_bigmem_cross`, X@address, y@address, as.integer(chunk_size))
      fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                   cross$XtX, cross$XtY, cross$x_means, cross$y_means,
                   as.integer(ncomp), tol)
      fit$mode <- "pls2"
      
      # Stream scores if requested (chunked T = (X - mu) %*% W)
      if (identical(scores, "big") || identical(scores, "r")) {
        local_sink <- NULL
        if (identical(scores, "big")) {
          if (is.null(scores_bm)) {
            local_sink <- bigmemory::filebacked.big.matrix(
              nrow = nrow(X), ncol = as.integer(fit$ncomp), type = "double",
              backingfile = if (is.null(scores_backingfile)) "scores.bin" else scores_backingfile,
              backingpath = if (is.null(scores_backingpath)) getwd() else scores_backingpath,
              descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile
            )
          } else if (inherits(scores_bm, "big.matrix")) {
            local_sink <- scores_bm
          } else if (inherits(scores_bm, "big.matrix.descriptor")) {
            local_sink <- bigmemory::attach.big.matrix(scores_bm)
          }
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
    .finalize_pls_fit(.post_scores(fit), "simpls")
  }
  
  run_bigmem_nipals <- function() {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    
    # allow numeric y → temporary big.matrix
    if (!inherits(y, "big.matrix")) {
      if (is.numeric(y)) {
        y_bm <- bigmemory::big.matrix(nrow = length(y), ncol = 1L, type = "double")
        y_bm[, 1] <- y
        y <- y_bm
      } else {
        stop("For backend='bigmem', y must be a big.matrix or numeric vector")
      }
    }
    
    if (mode == "pls1" && ncol(y) != 1L) stop("mode='pls1' requires y to have one column")
    
    sink_bm <- NULL
    if (identical(scores, "big") && scores_target == "existing") {
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
    
    if (mode == "pls1") {
      fit <- pls_streaming_bigmemory(
        X@address, y@address,
        as.integer(ncomp), as.integer(chunk_size),
        center = TRUE, scale = FALSE,
        tol = tol,
        return_big = identical(scores, "big")
      )
      fit$mode <- "pls1"
    } else {
      fit <- big_plsr_stream_fit_nipals(
        X@address, y@address, as.integer(ncomp),
        center = TRUE, scale = FALSE,
        chunk_size = as.integer(chunk_size),
        return_big = identical(scores, "big")
      )
      fit$mode <- "pls2"
    }
    
    # post scores routing
    if (identical(scores, "none")) {
      fit$scores <- NULL
    } else if (identical(scores, "r") && inherits(fit$scores, "big.matrix")) {
      fit$scores <- as.matrix(fit$scores[])
    } else if (identical(scores, "big")) {
      if (!is.null(sink_bm) && !is.null(fit$scores)) {
        if (inherits(fit$scores, "big.matrix")) {
          sink_bm[,] <- fit$scores[,]
        } else {
          sink_bm[,] <- fit$scores
        }
        fit$scores <- sink_bm
      } else if (!inherits(fit$scores, "big.matrix") && !is.null(fit$scores)) {
        bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(fit$scores), ncol = ncol(fit$scores), type = "double",
          backingfile = if (is.null(scores_backingfile)) "scores.bin" else scores_backingfile,
          backingpath = if (is.null(scores_backingpath)) getwd() else scores_backingpath,
          descriptorfile = if (is.null(scores_descriptorfile)) "scores.desc" else scores_descriptorfile
        )
        bm[,] <- fit$scores
        fit$scores <- bm
      }
    }
    
    .finalize_pls_fit(.post_scores(fit), "nipals")
  }
  
  # ---- Dispatch on algorithm -------------------------------------------------
  if (backend == "arma") {
    if (algo_in == "simpls") {
      return(run_dense_simpls())
    } else if (algo_in == "nipals") {
      return(run_dense_nipals())
    } else { # auto
      # Prefer SIMPLS; fall back to NIPALS if it errors
      out <- try(run_dense_simpls(), silent = TRUE)
      if (inherits(out, "try-error")) return(run_dense_nipals()) else return(out)
    }
  } else { # bigmem
    if (algo_in == "simpls") {
      return(run_bigmem_simpls())
    } else if (algo_in == "nipals") {
      return(run_bigmem_nipals())
    } else { # auto
      out <- try(run_bigmem_simpls(), silent = TRUE)
      if (inherits(out, "try-error")) return(run_bigmem_nipals()) else return(out)
    }
  }
}

.finalize_pls_fit <- function(fit, algorithm) {
  if (!is.list(fit)) return(fit)
  
  # Tag algorithm (preserve existing if already set)
  fit$algorithm <- algorithm %||% fit$algorithm
  
  ## ---- Normalize shapes / types (avoid touching big.matrix) ----
  if (!is.null(fit$coefficients) && !inherits(fit$coefficients, "big.matrix"))
    fit$coefficients <- as.matrix(fit$coefficients)
  if (!is.null(fit$intercept))
    fit$intercept <- as.numeric(fit$intercept)
  if (!is.null(fit$x_means))
    fit$x_means <- as.numeric(fit$x_means)
  if (!is.null(fit$y_means))
    fit$y_means <- as.numeric(fit$y_means)
  
  if (!is.null(fit$x_center)) {
    fit$x_center <- as.numeric(fit$x_center)
  } else if (!is.null(fit$x_means)) {
    fit$x_center <- fit$x_means
  }
  if (!is.null(fit$y_center)) {
    fit$y_center <- as.numeric(fit$y_center)
  } else if (!is.null(fit$y_means)) {
    fit$y_center <- fit$y_means
  }
  
  if (!is.null(fit$x_weights))  fit$x_weights  <- as.matrix(fit$x_weights)
  if (!is.null(fit$x_loadings)) fit$x_loadings <- as.matrix(fit$x_loadings)
  if (!is.null(fit$y_loadings)) fit$y_loadings <- as.matrix(fit$y_loadings)
  
  ## ---- Infer ncomp (prefer factor matrices) ----
  ncomp_from <- function(x) if (!is.null(x)) ncol(x) else NULL
  cand_ncomp <- ncomp_from(fit$x_weights) %||%
    ncomp_from(fit$x_loadings) %||%
    ncomp_from(fit$y_loadings)
  
  if (is.null(fit$ncomp)) {
    fit$ncomp <- cand_ncomp %||% fit$ncomp
  } else if (!is.null(cand_ncomp) && is.finite(fit$ncomp) && fit$ncomp != cand_ncomp) {
    # sync to matrices; silent adjust (or message() if you prefer)
    fit$ncomp <- cand_ncomp
  }
  if (is.null(fit$ncomp)) fit$ncomp <- 0L
  
  ## ---- Infer mode (pls1/pls2) if absent ----
  if (is.null(fit$mode)) {
    fit$mode <- {
      if (!is.null(fit$coefficients) && !inherits(fit$coefficients, "big.matrix")) {
        if (ncol(fit$coefficients) <= 1L) "pls1" else "pls2"
      } else if (!is.null(fit$intercept)) {
        if (length(fit$intercept) <= 1L) "pls1" else "pls2"
      } else if (!is.null(fit$y_loadings)) {
        # y_loadings: m x ncomp
        if (nrow(fit$y_loadings) <= 1L) "pls1" else "pls2"
      } else {
        # last resort: single-response default
        "pls1"
      }
    }
  }
  
  # Ensure class tag
  class(fit) <- unique(c("big_plsr", class(fit)))
  fit
}
