#' Unified PLS fit with auto backend and selectable algorithm
#'
#' @encoding UTF-8
#' 
#' @description
#' Dispatches to a dense (Arm/BLAS) backend for in-memory matrices
#' or to a streaming big.matrix backend when X (or Y) is a big.matrix.
#' Algorithm can be chosen between:
#' "simpls" (default), "nipals", "kernelpls", "widekernelpls",
#' "rkhs" (Rosipal & Trejo), "klogitpls", "sparse_kpls",
#' "rkhs_xy" (double RKHS), and "kf_pls" (Kalman-filter PLS, streaming).
#' 
#' The "kernelpls" paths now include a streaming XX'
#' variant for big.matrix inputs, with an optional row-chunking loop
#' controlled by \code{chunk_cols}.
#'
#' @param X numeric matrix or \code{bigmemory::big.matrix}
#' @param y numeric vector/matrix or \code{big.matrix}
#' @param ncomp number of latent components
#' @param tol numeric tolerance used in the core solver
#' @param backend one of \code{"auto"}, \code{"arma"}, \code{"bigmem"}
#' @param mode one of \code{"auto"}, \code{"pls1"}, \code{"pls2"}
#' @param algorithm one of \code{"auto"}, \code{"simpls"}, \code{"nipals"},
#'   \code{"kernelpls"}, \code{"widekernelpls"},
#'   \code{"rkhs"}, \code{"klogitpls"}, \code{"sparse_kpls"},
#'   \code{"rkhs_xy"}, \code{"kf_pls"}
#' @param scores one of \code{"none"}, \code{"r"}, \code{"big"}
#' @param chunk_size chunk size for the bigmem backend
#' @param chunk_cols columns chunk size for the bigmem backend
#' @param scores_name name for dense scores (or output big.matrix)
#' @param scores_target one of \code{"auto"}, \code{"new"}, \code{"existing"}
#' @param scores_bm optional existing big.matrix or descriptor for scores
#' @param scores_backingfile/backingpath/descriptorfile file-backed sink args
#' @param scores_colnames optional character vector for score column names
#' @param return_scores_descriptor logical; if TRUE and scores is big.matrix, add \code{$scores_descriptor}
#' @param coef_threshold Optional non-negative value used to hard-threshold
#'   the fitted coefficients after model estimation. When supplied, absolute
#'   coefficients strictly below the threshold are set to zero via
#'   [pls_threshold()].
#' @param kernel kernel name for RKHS/KPLS (\code{"linear"}, \code{"rbf"}, \code{"poly"}, \code{"sigmoid"})
#' @param gamma RBF/sigmoid/poly scale parameter
#' @param degree polynomial degree
#' @param coef0 polynomial/sigmoid bias
#' @param approx kernel approximation: \code{"none"}, \code{"nystrom"}, \code{"rff"}
#' @param approx_rank rank (columns / features) for the approximation
#' @param class_weights optional numeric weights for classes in \code{klogitpls}
#' @return a list with coefficients, intercept, weights, loadings, means,
#'   and optionally \code{$scores}.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r", algorithm = "simpls")
#' head(pls_predict_response(fit, X, ncomp = 2))
pls_fit <- function(
    X, y, ncomp,
    tol = 1e-8,
    backend = c("auto", "arma", "bigmem"),
    mode    = c("auto","pls1","pls2"),
    algorithm = c("auto","simpls","nipals",
                  "kernelpls","widekernelpls",
                  "rkhs","klogitpls","sparse_kpls","rkhs_xy","kf_pls"),
    scores  = c("none", "r", "big"),
    chunk_size = 10000L,
    chunk_cols = NULL,
    scores_name = "scores",
    scores_target = c("auto","new","existing"),
    scores_bm = NULL,
    scores_backingfile = NULL,
    scores_backingpath = NULL,
    scores_descriptorfile = NULL,
    scores_colnames = NULL,
    return_scores_descriptor = FALSE,
    coef_threshold = NULL,
    kernel = c("linear","rbf","poly","sigmoid"),
    gamma = 1.0,
    degree = 3L,
    coef0 = 0.0,
    approx = c("none","nystrom","rff"),
    approx_rank = NULL,
    class_weights = NULL    
) {
  backend   <- match.arg(backend)
  mode      <- match.arg(mode)
  scores    <- match.arg(scores)
  algo_in   <- match.arg(algorithm)
  scores_target <- match.arg(scores_target)
  kernel    <- match.arg(kernel)
  approx    <- match.arg(approx)
  if (!is.null(coef_threshold)) {
    if (!is.numeric(coef_threshold) || length(coef_threshold) != 1L || !is.finite(coef_threshold) || coef_threshold < 0) {
      stop("`coef_threshold` must be a single non-negative numeric value", call. = FALSE)
    }
  }
  
  # -------------------- Auto-selection helper --------------------
  .mem_bytes <- function() {
    gb <- getOption("bigPLSR.mem_budget_gb", 8)
    as.numeric(gb) * (1024^3)
  }
  .dims_of <- function(X) {
    if (inherits(X, "big.matrix")) c(nrow(X), ncol(X)) else c(NROW(X), NCOL(X))
  }
  
  .choose_algorithm_auto <- function(backend, X, y, ncomp) {
    is_big_local <- inherits(X, "big.matrix") || inherits(X, "big.matrix.descriptor")
    dims <- .dims_of(X); n <- as.integer(dims[1]); p <- as.integer(dims[2])
    M <- .mem_bytes()
    bytes <- 8
    need_XtX <- bytes * as.double(p) * as.double(p)      # bytes for p x p
    need_XXt <- bytes * as.double(n) * as.double(n)      # bytes for n x n
    can_XtX  <- need_XtX <= M
    can_XXt  <- need_XXt <= M
    shape_XtX <- (p <= 4L * n)
    shape_XXt <- (n <= 4L * p)
    if (can_XtX && shape_XtX) {
      algo_in <- "simpls"
    } else if (can_XXt && shape_XXt) {
      algo_in <- "widekernelpls"
    } else {
      algo_in <- "nipals"
    }
  }
  
  # If user asked for auto, pick algorithm now; explicit user choice always wins
  if (identical(algo_in, "auto")) {
    algo_in <- .choose_algorithm_auto(backend, X, y, ncomp)
  }
  
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
  
  .maybe_threshold <- function(fit) {
    if (is.null(coef_threshold)) {
      return(fit)
    }
    pls_threshold(fit, coef_threshold)
  }
  
  # ---- RKHS / kernel helpers (dense or bigmem will pass what they need) ----
  .rkhs_args <- list(kernel = kernel, gamma = gamma, degree = degree,
                     coef0 = coef0, approx = approx, approx_rank = approx_rank)
  
  # ---- DENSE BACKEND --------------------------------------------------------
  run_dense_simpls <- function() {
    if (is_big) {
      Xr <- as.matrix(X[])
      yr <- if (inherits(y, "big.matrix")) {
        if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[,1])
      } else {
        if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
      }
    } else {
      Xr <- as.matrix(X)
      yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
    }
    
    ## Always do SIMPLS on cross-products for parity with pls::simpls.fit
    Y <- if (is.matrix(yr)) yr else matrix(yr, nrow(Xr), 1L)
    x_means <- colMeans(Xr)
    y_means <- colMeans(Y)
    Xc <- sweep(Xr, 2L, x_means, FUN = "-")
    Yc <- sweep(Y,  2L, y_means, FUN = "-")
    XtX <- crossprod(Xc)          # p x p
    XtY <- crossprod(Xc, Yc)      # p x m
    fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                 XtX, XtY, x_means, y_means, as.integer(ncomp), tol)
    fit$mode <- if (ncol(Y) == 1L) "pls1" else "pls2"
    ## Ensure correct ncomp now (avoid fallback to coef dims later)
    if (is.null(fit$ncomp) || !is.finite(fit$ncomp) || fit$ncomp <= 0L) {
      if (!is.null(fit$x_weights)) fit$ncomp <- ncol(fit$x_weights)
    }
    
    ## Default to PLS-style scores: T = Xc %*% W %*% solve(P'W)
    if (scores != "none") {
      # internal escape hatch for developers:
      style <- getOption("bigPLSR.scores_style", "pls")  # "pls" or "raw" (undocumented)
      if (!is.null(fit$x_weights) && !is.null(fit$x_loadings)) {
        Rmat <- crossprod(fit$x_loadings, fit$x_weights)      # ncomp x ncomp
        Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
      } else {
        Rinv <- NULL
      }
      if (identical(style, "pls") && !is.null(Rinv)) {
        W_eff <- fit$x_weights %*% Rinv
      } else {
        W_eff <- fit$x_weights     # "raw" or fallback
      }
      
      Tmat <- Xc %*% W_eff
      if (identical(scores, "r")) {
        fit$scores <- Tmat
      } else {
        # scores=="big" → use sink or file-backed
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
      }} else {
        fit$scores <- NULL
      }
    fit <- .finalize_pls_fit(.post_scores(fit), "simpls")
    .maybe_threshold(fit)
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
    .maybe_threshold(fit)
  }
  
  run_dense_kernelpls <- function(kind) {
    if (is_big) {
      Xr <- as.matrix(X[])
      yr <- if (inherits(y, "big.matrix")) {
        if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[,1])
      } else {
        if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
      }
    } else {
      Xr <- as.matrix(X)
      yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
    }
    Xmat <- as.matrix(Xr)
    Ymat <- if (is.matrix(yr)) yr else matrix(yr, ncol = 1L)
    fit <- .Call(`_bigPLSR_cpp_kernel_pls`, Xmat, Ymat, as.integer(ncomp), tol, identical(kind, "wide"))
    fit$mode <- mode
    if (identical(scores, "none")) {
      fit$scores <- NULL
    } else if (identical(scores, "big") && !is.null(fit$scores)) {
      Tmat <- fit$scores
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
    }
    fit <- .finalize_pls_fit(.post_scores(fit), kind)
    .maybe_threshold(fit)
  }
  
  run_dense_rkhs <- function() {
    # Dense: build centered K, call dual RKHS-PLS core
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) as.matrix(y[, , drop = FALSE]) else as.matrix(y)
    # kernel + approx handled inside C++
    fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_dense`,
                 Xr, Yr, as.integer(ncomp), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 approx, if (is.null(approx_rank)) -1L else as.integer(approx_rank),
                 scores != "none")
    fit$mode <- if (ncol(Yr) == 1L) "pls1" else "pls2"
    # big scores sink if requested
    if (identical(scores, "big") && !is.null(fit$scores)) {
      Tmat <- as.matrix(fit$scores)
      if (inherits(scores_bm, "big.matrix")) {
        scores_bm[,] <- Tmat; fit$scores <- scores_bm
      } else if (inherits(scores_bm, "big.matrix.descriptor")) {
        bm <- bigmemory::attach.big.matrix(scores_bm); bm[,] <- Tmat; fit$scores <- bm
      }
    } else if (identical(scores, "none")) fit$scores <- NULL
    .finalize_pls_fit(.post_scores(fit), "rkhs")
  }
  
  run_dense_klogitpls <- function() {
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    yr <- if (inherits(y, "big.matrix")) as.numeric(y[,1]) else as.numeric(y)
    cw <- if (is.null(class_weights)) numeric() else as.numeric(class_weights)
    fit <- .Call(`_bigPLSR_cpp_klogit_pls_dense`,
                 Xr, yr, as.integer(ncomp), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 cw)
    fit$mode <- "pls1"
    .finalize_pls_fit(.post_scores(fit), "klogitpls")
  }
  
  run_dense_sparse_kpls <- function() {
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) as.matrix(y[, , drop = FALSE]) else as.matrix(y)
    fit <- .Call(`_bigPLSR_cpp_sparse_kpls_dense`,
                 Xr, Yr, as.integer(ncomp), as.numeric(tol))
    fit$mode <- if (ncol(Yr) == 1L) "pls1" else "pls2"
    .finalize_pls_fit(.post_scores(fit), "sparse_kpls")
  }
  
  run_dense_rkhs_xy <- function() {
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) as.matrix(y[, , drop = FALSE]) else as.matrix(y)
    fit <- .Call(`_bigPLSR_cpp_rkhs_xy_dense`,
                 Xr, Yr, as.integer(ncomp), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0))
    fit$mode <- if (ncol(Yr) == 1L) "pls1" else "pls2"
    .finalize_pls_fit(.post_scores(fit), "rkhs_xy")
  }
  
  # ---- BIGMEM BACKEND -------------------------------------------------------
  run_bigmem_simpls <- function() {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix")) stop("For backend='bigmem', y must be a big.matrix")
    
    ## Unified: always cross-products + SIMPLS (PLS1 or PLS2)
    cross <- .Call(`_bigPLSR_cpp_bigmem_cross`, X@address, y@address, as.integer(chunk_size))
    fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                 cross$XtX, cross$XtY, cross$x_means, cross$y_means,
                 as.integer(ncomp), tol)
    fit$mode <- if (ncol(cross$XtY) == 1L) "pls1" else "pls2"
    ## Ensure correct ncomp now (avoid fallback to coef dims later)
    if (is.null(fit$ncomp) || !is.finite(fit$ncomp) || fit$ncomp <= 0L) {
      if (!is.null(fit$x_weights)) fit$ncomp <- ncol(fit$x_weights)
    }
    
    # Prepare PLS-style weights for streaming: W_eff = W %*% solve(P'W)
    style <- getOption("bigPLSR.scores_style", "pls")  # hidden
    if (!is.null(fit$x_weights) && !is.null(fit$x_loadings)) {
      Rmat <- crossprod(fit$x_loadings, fit$x_weights)   # ncomp x ncomp
      Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
    } else {
      Rinv <- NULL
    }
    if (identical(style, "pls") && !is.null(Rinv)) {
      W_eff <- fit$x_weights %*% Rinv
    } else {
      W_eff <- fit$x_weights
    }
    
    # Stream scores if requested: T = (X - mu) %*% W_eff
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
                          X@address, W_eff, fit$x_means,
                          as.integer(chunk_size),
                          if (is.null(local_sink)) NULL else local_sink,
                          identical(scores, "big"))
    } else {
      fit$scores <- NULL
    }
    fit <- .finalize_pls_fit(.post_scores(fit), "simpls")
    .maybe_threshold(fit)
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
    
    fit <- .finalize_pls_fit(.post_scores(fit), "nipals")
    .maybe_threshold(fit)
  }
  
  run_bigmem_kernelpls <- function(kind) {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    gram_mode <- getOption("bigPLSR.kpls_gram", "cols")
    use_rows  <- identical(gram_mode, "rows")
    if (is.null(chunk_cols)) chunk_cols_loc <- max(1024L, as.integer(0.1 * nrow(X))) else chunk_cols_loc <- as.integer(chunk_cols)
    if (identical(gram_mode, "auto")) {
      use_rows <- (nrow(X) > 4L * ncol(X))
    }
    if (use_rows) {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_xxt`,
                   X@address, y@address,
                   as.integer(ncomp), as.integer(chunk_size), as.integer(chunk_cols_loc),
                   TRUE,
                   identical(scores, "big"))
    } else {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_cols`,
                   X@address, y@address,
                   as.integer(ncomp), as.integer(chunk_size),
                   TRUE,
                   identical(scores, "big"))
    }
    fit$mode <- if (ncol(fit$coefficients) <= 1L) "pls1" else "pls2"
    if (identical(scores, "r") && inherits(fit$scores, "big.matrix")) {
      fit$scores <- as.matrix(fit$scores[])
    }
    fit <- .finalize_pls_fit(.post_scores(fit), "kernelpls")
    .maybe_threshold(fit)
  }
  
  run_bigmem_rkhs <- function() {
    if (!inherits(X, "big.matrix") || !inherits(y, "big.matrix"))
      stop("For backend='bigmem' and algorithm='rkhs', both X and y must be big.matrix")
    fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_bigmem`,
                 X@address, y@address,
                 as.integer(ncomp), as.integer(chunk_size), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 approx, if (is.null(approx_rank)) -1L else as.integer(approx_rank),
                 scores != "none")
    # bigmem: scores returned as big or dense depending on request inside C++
    .finalize_pls_fit(.post_scores(fit), "rkhs")
  }
  
  run_bigmem_klogitpls <- function() {
    if (!inherits(X, "big.matrix")) stop("X must be big.matrix for bigmem klogitpls")
    yv <- if (inherits(y, "big.matrix")) as.numeric(y[,1]) else as.numeric(y)
    cw <- if (is.null(class_weights)) numeric() else as.numeric(class_weights)
    fit <- .Call(`_bigPLSR_cpp_klogit_pls_bigmem`,
                 X@address, yv,
                 as.integer(ncomp), as.integer(chunk_size), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 cw)
    .finalize_pls_fit(.post_scores(fit), "klogitpls")
  }
  
  run_bigmem_kf_pls <- function() {
    if (!inherits(X, "big.matrix") || !inherits(y, "big.matrix"))
      stop("For backend='bigmem' and algorithm='kf_pls', both X and y must be big.matrix")
    fit <- .Call(`_bigPLSR_cpp_kf_pls_stream`,
                 X@address, y@address,
                 as.integer(ncomp), as.integer(chunk_size), as.numeric(tol))
    .finalize_pls_fit(.post_scores(fit), "kf_pls")
  }
  
  run_bigmem_kernelpls <- function(kind) {
    # Keep legacy kernelpls wiring via dense helper on pulled blocks for now
    return(run_dense_kernelpls(kind))
  }
  
  # ---- Dispatch on algorithm -------------------------------------------------
  if (backend == "arma") {
    if (algo_in == "simpls") {
      return(run_dense_simpls())
    } else if (algo_in == "nipals") {
      return(run_dense_nipals())
    } else if (algo_in == "kernelpls") {
      return(run_dense_kernelpls("kernel"))
    } else if (algo_in == "widekernelpls") {
      return(run_dense_kernelpls("wide"))
    } else if (algo_in == "rkhs") {
      return(run_dense_rkhs())
    } else if (algo_in == "klogitpls") {
      return(run_dense_klogitpls())
    } else if (algo_in == "sparse_kpls") {
      return(run_dense_sparse_kpls())
    } else if (algo_in == "rkhs_xy") {
      return(run_dense_rkhs_xy())
    } else { # auto
      out <- try(run_dense_simpls(), silent = TRUE)
      if (inherits(out, "try-error")) {
        out <- try(run_dense_kernelpls("kernel"), silent = TRUE)
        if (inherits(out, "try-error")) {
          out <- try(run_dense_kernelpls("wide"), silent = TRUE)
        }
        if (inherits(out, "try-error")) return(run_dense_nipals()) else return(out)
      }
      return(out)
    }
  } else { # bigmem
    if (algo_in == "simpls") {
      return(run_bigmem_simpls())
    } else if (algo_in == "nipals") {
      return(run_bigmem_nipals())
    } else if (algo_in == "kernelpls") {
      return(run_bigmem_kernelpls("kernel"))
    } else if (algo_in == "widekernelpls") {
      return(run_bigmem_kernelpls("wide"))
    } else if (algo_in == "rkhs") {
      return(run_bigmem_rkhs())
    } else if (algo_in == "klogitpls") {
      return(run_bigmem_klogitpls())
    } else if (algo_in == "kf_pls") {
      return(run_bigmem_kf_pls())
    } else { # auto
      out <- try(run_bigmem_simpls(), silent = TRUE)
      if (inherits(out, "try-error")) {
        out <- try(run_bigmem_kernelpls("kernel"), silent = TRUE)
        if (inherits(out, "try-error")) {
          out <- try(run_bigmem_kernelpls("wide"), silent = TRUE)
        }
        if (inherits(out, "try-error")) return(run_bigmem_nipals()) else return(out)
      }
      return(out)
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
  if (!is.null(cand_ncomp)) {
    # always trust factor matrices; override mismatched or missing ncomp
    if (is.null(fit$ncomp) || !is.finite(fit$ncomp) || fit$ncomp != cand_ncomp) {
      fit$ncomp <- cand_ncomp
    }
  } else if (is.null(fit$ncomp)) {
    # last resort if nothing else is available
    if (!is.null(fit$coefficients) && !inherits(fit$coefficients, "big.matrix")) {
      fit$ncomp <- 1L  # safest conservative default (PLS1 coef has 1 column)
    } else {
      fit$ncomp <- 0L
    }
  }
  
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
