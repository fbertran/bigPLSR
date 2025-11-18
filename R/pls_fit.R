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
#' @param scores_backingfile Character; file name for file-backed scores (when `scores="big"`).
#' @param scores_backingpath  Character; directory for the file-backed scores.
#'   Defaults to `getwd()` or `tempdir()` in streamed predict, unless overridden.
#' @param scores_descriptorfile Character; descriptor file name for the file-backed scores.
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
#  gamma    
#  degree   
#  coef0    
  approx    <- match.arg(approx)
  if (!is.null(coef_threshold)) {
    if (!is.numeric(coef_threshold) || length(coef_threshold) != 1L || !is.finite(coef_threshold) || coef_threshold < 0) {
      stop("`coef_threshold` must be a single non-negative numeric value", call. = FALSE)
    }
  }
  
  # sensible defaults for RKHS-XY
  kernel_x   <- getOption("bigPLSR.rkhs_xy.kernel_x",   "rbf")
  gamma_x    <- getOption("bigPLSR.rkhs_xy.gamma_x",    1 / max(1L, ncol(as.matrix(X))))
  degree_x   <- getOption("bigPLSR.rkhs_xy.degree_x",   3L)
  coef0_x    <- getOption("bigPLSR.rkhs_xy.coef0_x",    1.0)
  kernel_y   <- getOption("bigPLSR.rkhs_xy.kernel_y",   "linear")
  gamma_y    <- getOption("bigPLSR.rkhs_xy.gamma_y",    1 / max(1L, NCOL(as.matrix(y))))
  degree_y   <- getOption("bigPLSR.rkhs_xy.degree_y",   3L)
  coef0_y    <- getOption("bigPLSR.rkhs_xy.coef0_y",    1.0)
  lambda_x   <- getOption("bigPLSR.rkhs_xy.lambda_x",   1e-6)
  lambda_y   <- getOption("bigPLSR.rkhs_xy.lambda_y",   1e-6)
  
  # hidden knobs for klogitpls (also readable via options)
  klogit_maxit   <- getOption("bigPLSR.klogitpls.max_irls_iter", 50L)
  klogit_alt     <- getOption("bigPLSR.klogitpls.alt_updates",    0L)
  klogit_tol     <- getOption("bigPLSR.klogitpls.tol_irls",       1e-8)

  # KF-PLS knobs (default values; can be overridden in options)
  kf_lambda <- getOption("bigPLSR.kf.lambda", 0.995)
  kf_qproc  <- getOption("bigPLSR.kf.q_proc", 1e-6)
  
  # ---- class_weights: DO NOT use match.arg() here ---------------------------
  # Accept: numeric length-2 vector c(w0, w1), or named vector with class names.
  normalize_class_weights <- function(y_vec, cw) {
    if (is.null(cw)) return(numeric(0))
    if (!is.numeric(cw)) {
      stop("class_weights must be a numeric vector of length 2 (w0, w1) or a named numeric vector keyed by class labels.", call. = FALSE)
    }
    if (length(cw) != 2L) {
      stop("class_weights must have length 2: c(weight_for_class0, weight_for_class1).", call. = FALSE)
    }
    # If y is factor with 2 levels, allow names to map to levels
    if (is.factor(y_vec)) {
      lev <- levels(y_vec)
      if (length(lev) != 2L) stop("klogitpls requires a binary response (factor with 2 levels).", call. = FALSE)
      if (!is.null(names(cw)) && all(lev %in% names(cw))) {
        cw <- as.numeric(cw[lev])
      } else if (!is.null(names(cw)) && all(c("0","1") %in% names(cw))) {
        cw <- as.numeric(cw[c("0","1")])
      } else {
        # Unnamed numeric vector assumed order is c(w0, w1) for lev[1], lev[2]
        cw <- as.numeric(cw)
      }
    } else {
      # y numeric → assume 0/1 mapping; order c(w0, w1)
      cw <- as.numeric(cw)
    }
    cw
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
  
  # ---- SIMPLS projection solver selection & helpers -------------------------
  .simpls_solver <- function() {
    # Allow tests to select the linear solve used to form W %*% solve(R)
    # Supported: "chol" (default, fast), "tri", "qr", "solve"
    m <- getOption("bigPLSR.simpls.solve", "chol")
    match.arg(tolower(m), c("chol", "tri", "qr", "solve"))
  }
  
  .is_upper_tri <- function(M, tol = 1e-12) {
    # Cheap check for (near) upper-triangular structure
    if (!is.matrix(M)) return(FALSE)
    if (nrow(M) != ncol(M)) return(FALSE)
    all(abs(M[lower.tri(M)]) <= tol)
  }
  
  # Small helper: maybe build centered X only when scores are actually needed
  .bigPLSR_center_if_needed <- function(X, mu, need_scores) {
    if (!isTRUE(need_scores)) return(NULL)     # skip allocation entirely
    if (is.null(X) || is.null(mu)) return(NULL)
    sweep(X, 2L, mu, FUN = "-")
  }
  # Compute B %*% solve(A) without forming solve(A), honoring solver option.
  # A is typically R = t(P) %*% W  (ncomp x ncomp), B = W (p x ncomp).
  .simpls_right_solve <- function(A, B, method = .simpls_solver()) {
    p <- nrow(B); k <- ncol(B)
    stopifnot(nrow(A) == ncol(A), ncol(A) == k)
    meth <- method
    if (meth == "tri" || (meth == "chol" && .is_upper_tri(A))) {
      # Fast path: R is (nearly) upper-triangular → use backsolve
      # X = B %*% R^{-1}  ==  t( backsolve(R, t(B), upper.tri = TRUE, transpose = FALSE) )
      return(t(backsolve(A, t(B), upper.tri = TRUE, transpose = FALSE)))
    }
    if (meth == "chol") {
      # If not triangular, try normal-equations Cholesky on A'A (square/invertible → OK).
      # Solve X = B %*% A^{-1} by solving (A'A) Z = A' B' for Z = (A^{-T}) B'
      # then X = t(Z) = B %*% A^{-1}. This is stable if A is well-conditioned.
      AtA <- crossprod(A)      # k x k, SPD if A full rank
      Rch <- try(chol(AtA), silent = TRUE)
      if (!inherits(Rch, "try-error")) {
        rhs <- t(A) %*% t(B)                  # k x p
        Y   <- forwardsolve(t(Rch), rhs)      # solve R' Y = rhs
        Z   <- backsolve(Rch, Y)              # solve R Z  = Y
        return(t(Z))                          # X = t(Z)
      }
      # Fallback if Cholesky fails
      meth <- "solve"
    }
    if (meth == "qr") {
      # Solve X A = B ⇒ X = B %*% A^{-1}; implement as X^T = solve(t(A), t(B))
      return(t(qr.solve(t(A), t(B))))
    }
    # Generic fallback
    B %*% solve(A)
  }
  
  # ---- DENSE BACKEND --------------------------------------------------------
  run_dense_simpls <- function() {
    # ---- Dense inputs
    if (is_big) {
      Xr <- as.matrix(X[])
      yr <- if (inherits(y, "big.matrix")) {
        if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[, 1])
      } else {
        if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
      }
    } else {
      Xr <- as.matrix(X)
      yr <- if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
    }
    
    # ---- Build cross-products WITHOUT materializing Xc/Yc (major speedup)
    # Xc'Xc = X'X − n * (mu_x %*% t(mu_x))
    # Xc'Yc = X'Y − n * (mu_x %*% t(mu_y))
    Y <- if (is.matrix(yr)) yr else matrix(yr, nrow(Xr), 1L)
    cross_mode <- .bigPLSR_simpls_cross_mode()
    if (identical(cross_mode, "dense_cxx") && exists("cpp_dense_cross", mode = "function")) {
      # Fast path: fused centering + cross-products in C++ (no big Xc/Yc temporaries)
      use_syrk <- isTRUE(getOption("bigPLSR.simpls.use_syrk", TRUE))
      cr <- cpp_dense_cross(Xr, Y, use_syrk)      
      x_means <- as.numeric(cr$x_means)
      y_means <- as.numeric(cr$y_means)
      XtX     <- cr$XtX
      XtY     <- cr$XtY
      # no Xc needed here; scores will use Xr %*% W_eff - shift
    } else {
      # R fallback: explicit centering + crossprod
      x_means <- colMeans(Xr)
      y_means <- colMeans(Y)
      need_scores <- !identical(scores, "none")
      Xc <- .bigPLSR_center_if_needed(Xr, x_means, need_scores || TRUE)  # we do need Xc for XtX/XtY here
      Yc <- sweep(Y,  2L, y_means, FUN = "-")
      XtX <- crossprod(Xc)          # p x p
      XtY <- crossprod(Xc, Yc)      # p x m
    }
    
    fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                 XtX, XtY, x_means, y_means, as.integer(ncomp), tol)
    fit$mode <- if (ncol(Y) == 1L) "pls1" else "pls2"
    ## Ensure correct ncomp now (avoid fallback to coef dims later)
    if (is.null(fit$ncomp) || !is.finite(fit$ncomp) || fit$ncomp <= 0L) {
      if (!is.null(fit$x_weights)) fit$ncomp <- ncol(fit$x_weights)
    }
    ## Default to PLS-style scores, without materialising Xc:
    ##   T = (X - mu_x) %*% W_eff = X %*% W_eff - 1 %*% (mu_x %*% W_eff)
    ## Avoid any work if not asked:
    if (!identical(scores, "none")) {
      style  <- getOption("bigPLSR.scores_style", "pls")  # "pls" or "raw" (hidden)
      solver <- .simpls_solver()
      if (!is.null(fit$x_weights) && !is.null(fit$x_loadings) && identical(style, "pls")) {
        Rmat  <- crossprod(fit$x_loadings, fit$x_weights)      # k x k, k = ncomp
        W_eff <- .simpls_right_solve(Rmat, fit$x_weights, method = solver)
      } else {
        W_eff <- fit$x_weights  # "raw" or missing factors
      }
      # Scores without Xc materialization: T = X %*% W_eff  -  1·(mu_x %*% W_eff)
      # C++ fused path: T = X W_eff - 1 (mu_x W_eff)
      if (exists("cpp_dense_scores", mode = "function")) {
        Tmat <- cpp_dense_scores(Xr, x_means, W_eff)
      } else {
        TW0   <- Xr %*% W_eff
        shift <- as.numeric(x_means %*% W_eff)
        # fast column shift:
        for (j in seq_len(ncol(TW0))) TW0[, j] <- TW0[, j] - shift[j]
        Tmat <- TW0
      }
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
          # CRAN: never write outside tempdir() by default
          default_path <- getOption("bigPLSR.backingpath_default", tempdir())
          bm <- bigmemory::filebacked.big.matrix(
            nrow = nrow(Tmat), ncol = ncol(Tmat), type = "double",
            backingfile = scores_backingfile,
            backingpath = scores_backingpath %||% default_path,
            descriptorfile = scores_descriptorfile %||% "scores.desc"
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
    fit <- .finalize_pls_fit(.post_scores(fit), "simpls")
    if (.bigPLSR_should_store_X(X)) fit$X <- Xr
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
        if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    fit$mode <- mode
    
    # Ensure means are present (needed for center-free scoring below)
    if (is.null(fit$x_means)) fit$x_means <- as.numeric(colMeans(Xr))
    
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
        # CRAN: never write outside tempdir() by default
        default_path <- getOption("bigPLSR.backingpath_default", tempdir())
        bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(Tmat), ncol = ncol(Tmat), type = "double",
          backingfile = scores_backingfile,
          backingpath = scores_backingpath %||% default_path,
          descriptorfile = scores_descriptorfile %||% "scores.desc"
        )
        bm[,] <- Tmat
        fit$scores <- bm
      }
    } else if (identical(scores, "none")) {
      fit$scores <- NULL
    } else if (is.null(fit$scores)) {
    # ---- Center-free score formation for NIPALS (no Xc materialization)
    # Style parity with SIMPLS: optionally orthonormalize via M = solve(P'W)
    style <- getOption("bigPLSR.scores_style", "pls")
    if (!is.null(fit$x_weights) && !is.null(fit$x_loadings)) {
      Rmat <- crossprod(fit$x_loadings, fit$x_weights)
      Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
    } else {
      Rinv <- NULL
    }
    if (identical(style, "pls") && !is.null(Rinv)) {
      W_eff <- fit$x_weights %*% Rinv
    } else {
      W_eff <- fit$x_weights
    }
    TW0   <- Xr %*% W_eff
    shift <- as.numeric(fit$x_means %*% W_eff)
    Tmat  <- sweep(TW0, 2L, shift, FUN = "-")
    if (identical(scores, "r")) {
      fit$scores <- Tmat
    } else {
      # scores == "big"
      if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix")) {
        scores_bm[,] <- Tmat; fit$scores <- scores_bm
      } else if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix.descriptor")) {
        bm <- bigmemory::attach.big.matrix(scores_bm); bm[,] <- Tmat; fit$scores <- bm
      } else if (!is.null(scores_backingfile)) {
        # CRAN: never write outside tempdir() by default
        default_path <- getOption("bigPLSR.backingpath_default", tempdir())
        bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(Tmat), ncol = ncol(Tmat), type = "double",
          backingfile = scores_backingfile,
          backingpath = scores_backingpath %||% default_path,
          descriptorfile = scores_descriptorfile %||% "scores.desc"
        )
        bm[,] <- Tmat; fit$scores <- bm
      } else {
        fit$scores <- Tmat
      }
    }
  }
    
    fit <- .finalize_pls_fit(.post_scores(fit), "nipals")
    if (isTRUE(return_scores_descriptor) && inherits(fit$scores, "big.matrix")) {
      fit$scores_descriptor <- bigmemory::describe(fit$scores)
    }
        if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
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
    # persist kernel params for prediction
    fit$kernel_x <- fit$kernel <- if (kind == "wide") "widekernelpls" else "kernelpls"
    # heuristics: we still need actual kernel hyperparams if non-linear is used
    fit$gamma_x  <- getOption("bigPLSR.kernel.gamma", 1 / ncol(Xr))
    fit$degree_x <- getOption("bigPLSR.kernel.degree", 3L)
    fit$coef0_x  <- getOption("bigPLSR.kernel.coef0", 1)
    # store X only if small enough
    if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    # store K-centering stats when feasible (lets predict() avoid needing X)
    fit <- .bigPLSR_try_store_kstats(fit, Xr,
                                     kernel = "rbf", gamma = fit$gamma_x,
                                     degree = fit$degree_x, coef0 = fit$coef0_x)
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
        # CRAN: never write outside tempdir() by default
        default_path <- getOption("bigPLSR.backingpath_default", tempdir())
        bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(Tmat), ncol = ncol(Tmat), type = "double",
          backingfile = scores_backingfile,
          backingpath = scores_backingpath %||% default_path,
          descriptorfile = scores_descriptorfile %||% "scores.desc"
        )
        bm[,] <- Tmat
        fit$scores <- bm
      }
    }
    fit <- .finalize_pls_fit(.post_scores(fit), kind)
        if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    .maybe_threshold(fit)
  }
  
  run_dense_rkhs <- function() {
    # Dense: build centered K, call dual RKHS-PLS core
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) as.matrix(y[, , drop = FALSE]) else as.matrix(y)
    # kernel + approx handled inside C++
    fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_dense`, Xr, Yr, 
                 as.integer(ncomp), as.numeric(tol),
                 kernel, as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 approx,
                 if (is.null(approx_rank)) -1L else as.integer(approx_rank))
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
        if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    .finalize_pls_fit(.post_scores(fit), "rkhs")
  }
  
  run_dense_sparse_kpls <- function() {
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) as.matrix(y[, , drop = FALSE]) else as.matrix(y)
    fit <- .Call(`_bigPLSR_cpp_sparse_kpls_dense`,
                 Xr, Yr, as.integer(ncomp), as.numeric(tol))
    fit$mode <- if (ncol(Yr) == 1L) "pls1" else "pls2"
        if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    .finalize_pls_fit(.post_scores(fit), "sparse_kpls")
  }
  
  run_dense_rkhs_xy <- function() {
    # Dense Double RKHS (X and Y in RKHS)
    Xr <- as.matrix(if (is.matrix(X)) X else if (inherits(X,"big.matrix")) X[] else X)
    Yr <- if (is.matrix(y)) y else if (inherits(y,"big.matrix")) y[] else as.matrix(y)
    fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_xy_dense`,
                 Xr, Yr, as.integer(ncomp), as.numeric(tol),
                 as.character(kernel_x), as.numeric(gamma_x), as.integer(degree_x), as.numeric(coef0_x),
                 as.character(kernel_y), as.numeric(gamma_y), as.integer(degree_y), as.numeric(coef0_y),
                 as.numeric(lambda_x), as.numeric(lambda_y))
    fit$mode <- if (NCOL(Yr) == 1L) "pls1" else "pls2"
    if (.bigPLSR_should_store_X(X)) fit$X <- Xr
    fit$algorithm <- "rkhs_xy"
    # persist or default kernel hyperparams (X side) for prediction
    kx <- tolower(fit$kernel_x %||% getOption("bigPLSR.rkhs_xy.kernel_x", "rbf"))
    gx <- fit$gamma_x %||% getOption("bigPLSR.rkhs_xy.gamma_x", 1 / ncol(Xr))
    dx <- fit$degree_x %||% getOption("bigPLSR.rkhs_xy.degree_x", 3L)
    c0x<- fit$coef0_x  %||% getOption("bigPLSR.rkhs_xy.coef0_x", 1)
    fit <- .bigPLSR_try_store_kstats(fit, Xr,
                                     kernel = kx, gamma = gx, degree = dx, coef0 = c0x,
                                     force = TRUE)
    # ALWAYS store X-kernel centering stats (double RKHS needs robust centering)
    fit$kernel_x <- kx; fit$gamma_x <- gx; fit$degree_x <- dx; fit$coef0_x <- c0x
    # store X only if small enough (helps precompute cross-kernels quickly)
    if (.bigPLSR_should_store_X(Xr)) fit$X <- Xr
    .finalize_pls_fit(.post_scores(fit), "rkhs_xy")
  }
  
  .sigmoid <- function(eta) {
    p <- 1 / (1 + exp(-eta))
    pmin(pmax(p, 1e-12), 1 - 1e-12)
  }
  
  .irls_binomial <- function(T, y, w_class = NULL, maxit = 50L, tol = 1e-8) {
    y <- as.numeric(y > 0.5)
    n <- nrow(T); A <- ncol(T)
    M <- cbind(1, T)
    theta <- numeric(A + 1L)
    converged <- FALSE
    it <- 0L
    for (it in seq_len(maxit)) {
      eta <- drop(M %*% theta)
      p   <- .sigmoid(eta)
      w   <- p * (1 - p)
      if (!is.null(w_class) && length(w_class) >= 2L) {
        w <- w * ifelse(y > 0.5,
                        w_class[[2]] %||% w_class[["1"]] %||% w_class[[2]],
                        w_class[[1]] %||% w_class[["0"]] %||% w_class[[1]])
      }
      z <- eta + (y - p) / pmax(w, 1e-12)
      W <- sqrt(w)
      MW <- M * W
      zW <- z * W
      coef <- tryCatch(
        solve(crossprod(MW), crossprod(MW, zW)),
        error = function(e) NULL
      )
      if (is.null(coef)) break
      step <- max(abs(coef - theta))
      theta <- coef
      if (step < tol) {
        converged <- TRUE
        break
      }
    }
    list(
      beta = if (A > 0L) theta[-1L] else numeric(),
      b = theta[1L],
      fitted = .sigmoid(drop(M %*% theta)),
      iter = it,
      converged = converged
    )
  }
  
  .run_irls <- function(T, ybin, cw, maxit, tol) {
    w_class <- if (length(cw) == 0L) NULL else cw
    if (exists("cpp_irls_binomial", mode = "function")) {
      cpp_irls_binomial(T, ybin, w_class = w_class, maxit = maxit, tol = tol)
    } else {
      .irls_binomial(T, ybin, w_class = w_class, maxit = maxit, tol = tol)
    }
  }
  
  .calibrate_weighted_logit <- function(T, ybin, ir_w, cw, maxit, tol) {
    if (length(cw) == 0L) return(ir_w)
    pos <- ybin > 0.5
    neg <- !pos
    if (!any(pos) || !any(neg)) return(ir_w)
    
    base <- .run_irls(T, ybin, numeric(0), maxit, tol)
    design <- cbind(1, T)
    eta_w <- drop(design %*% c(ir_w$b, ir_w$beta))
    eta_base <- drop(design %*% c(base$b, base$beta))
    p_neg_w <- .sigmoid(eta_w[neg])
    p_neg_base <- .sigmoid(eta_base[neg])
    p_pos_w <- .sigmoid(eta_w[pos])
    p_pos_base <- .sigmoid(eta_base[pos])
    if (mean(p_neg_w) <= mean(p_neg_base) + 1e-8) return(ir_w)
    
    find_root <- function(fun) {
      upper <- 1
      val <- fun(upper)
      iter <- 0L
      while (is.finite(val) && val > 0 && iter < 25L) {
        upper <- upper * 2
        val <- fun(upper)
        iter <- iter + 1L
      }
      if (!is.finite(val) || val > 0) return(NA_real_)
      tryCatch(stats::uniroot(fun, lower = 0, upper = upper)$root, error = function(e) NA_real_)
    }
    
    root_neg <- find_root(function(delta) mean(.sigmoid(eta_w[neg] - delta)) - mean(p_neg_base))
    if (!is.finite(root_neg) || root_neg <= 0) return(ir_w)
    
    root_pos <- if (mean(p_pos_w) > mean(p_pos_base) + 1e-8) {
      find_root(function(delta) mean(.sigmoid(eta_w[pos] - delta)) - mean(p_pos_base))
    } else {
      NA_real_
    }
    
    delta <- root_neg
    if (is.finite(root_pos) && root_pos > 0) {
      delta <- min(delta, 0.999 * root_pos)
    }
    if (!is.finite(delta) || delta <= 0) return(ir_w)
    
    ir_w$b <- ir_w$b - delta
    ir_w$fitted <- .sigmoid(eta_w - delta)
    ir_w
  }
  
  # ---- Kernel RKHS Logistic PLS (dense) -----------------------
  run_dense_klogitpls <- function() {
    # inputs
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    yv <- if (is.matrix(y)) as.numeric(y[, 1]) else if (inherits(y, "big.matrix")) as.numeric(y[,1]) else as.numeric(y)
    # Prepare binary y and classes
    if (is.factor(yv)) {
      lev  <- levels(yv)
      if (length(lev) != 2L) stop("klogitpls requires a binary response (factor with 2 levels).", call. = FALSE)
      ybin <- as.numeric(yv == lev[2L])
      classes <- lev
    } else {
      ybin <- as.numeric(as.numeric(yv) > 0.5)
      classes <- c("0","1")
    }
    cw <- normalize_class_weights(if (is.factor(y)) y else factor(ybin, levels = c(0,1), labels = classes), class_weights)

    # hyper-params (allow ... overrides if present, else options, else defaults)
    kernel <- kernel %||% getOption("bigPLSR.klogitpls.kernel", "rbf")
    gamma  <- gamma %||% getOption("bigPLSR.klogitpls.gamma",  1 / ncol(Xr))
    degree <- degree %||% getOption("bigPLSR.klogitpls.degree", 3L)
    coef0  <- coef0 %||% getOption("bigPLSR.klogitpls.coef0",  1.0)
    # IRLS controls / class weights
#    cw     <- if (exists("class_weights", inherits = FALSE) && !is.null(class_weights)) as.numeric(class_weights) else numeric()
    itmax  <- if (exists("klogit_maxit", inherits = FALSE)) klogit_maxit else getOption("bigPLSR.klogitpls.maxit", 50L)
    ittol  <- if (exists("klogit_tol",   inherits = FALSE)) klogit_tol   else getOption("bigPLSR.klogitpls.tol",   1e-8)
    alt_it <- if (exists("klogit_alt",   inherits = FALSE)) klogit_alt   else getOption("bigPLSR.klogitpls.alt",   0L)
    u_rdg  <- getOption("bigPLSR.klogitpls.u_ridge", 1e-10)
    
    # ----- DENSE: build Gram, do KPLS (scores), build u_basis, IRLS -----
    K      <- .bigPLSR_make_kernel(Xr, Xr, kernel, gamma, degree, coef0)
    
    # --- Centering Train (HKH): stats + centered Gram ---
    r_train <- colMeans(K)
    g_train <- mean(K)
    Kc      <- .bigPLSR_center_cross_kernel(K, r_train = r_train, g_train = g_train)

    # KPLS scores from Gram (centered inside C++; if not, compute T then still OK)
    kfit0  <- cpp_kpls_from_gram(Kc, matrix(ybin, ncol = 1L), as.integer(ncomp), tol)
    T      <- as.matrix(kfit0$scores)
    A      <- ncol(T)
    if (A < 1L) stop("klogitpls: no component extracted")
    uB     <- as.matrix(kfit0$u_basis)
#    U      <- tryCatch(solve(Kc + diag(u_rdg, nrow(Kc)), T),
#                   error = function(e) MASS::ginv(Kc + diag(u_rdg, nrow(Kc))) %*% T)
    # IRLS on scores
    ir <- if (exists('cpp_irls_binomial', mode = 'function')) {
      cpp_irls_binomial(T, ybin, w_class = cw, maxit = itmax, tol = ittol)
    } else {
      ir <- .run_irls(T, ybin, cw, itmax, ittol)
    }
    ir <- .run_irls(T, ybin, cw, itmax, ittol)
    # optional alternations
    if (alt_it > 0L) {
      for (kk in seq_len(alt_it)) {
        eta <- drop(cbind(1, T) %*% c(ir$b, ir$beta))
        p   <- .sigmoid(eta)
        # Re-use **Kc** (centered Gram train)
        kfitk <- cpp_kpls_from_gram(Kc, matrix(p, ncol = 1L), as.integer(ncomp), tol)
        T  <- as.matrix(kfitk$scores)
        uB <- as.matrix(kfitk$u_basis)
        ir <- if (exists('cpp_irls_binomial', mode = 'function')) {
          cpp_irls_binomial(T, ybin, w_class = cw, maxit = itmax, tol = ittol)
        } else {
          ir <- .run_irls(T, ybin, cw, itmax, ittol)
        }
      }
    }
    ir <- .calibrate_weighted_logit(T, ybin, ir, cw, itmax, ittol)
    obj <- list(
      algorithm    = "klogitpls",
      family       = "binomial",
      ncomp        = A,
      classes      = classes,
      intercept    = ir$b,
      latent_coef  = ir$beta,
      class_weights = if (length(cw)) cw else NULL,
      u_basis      = uB %||% NULL,
      # store kernel params & centering stats for predict
      kernel       = kernel, gamma = gamma, degree = degree, coef0 = coef0,
      kstats_x     = list(r = r_train, g = g_train),
      X            = if (.bigPLSR_should_store_X(Xr)) Xr else NULL,
      scores       = T,
      mode         = "pls1"
    )
    return(.finalize_pls_fit(.post_scores(obj), "klogitpls"))
  }
  
  # ---- KF-PLS (EWMA cross-products, SIMPLS extraction) ----------------------
  run_dense_kf_pls <- function() {
    Xr <- if (is_big) as.matrix(X[]) else as.matrix(X)
    Yr <- if (inherits(y, "big.matrix")) {
      if (mode == "pls2" && ncol(y) > 1L) as.matrix(y[, , drop = FALSE]) else as.numeric(y[,1])
    } else {
      if (mode == "pls2" && is.matrix(y) && ncol(y) > 1L) as.matrix(y) else as.numeric(y)
    }
    fit <- cpp_kf_pls_dense(Xr, Yr, as.integer(ncomp),
                            tol = tol, lambda = kf_lambda, q_proc = kf_qproc)
    fit$mode <- if (is.matrix(Yr) && ncol(Yr) > 1L) "pls2" else "pls1"
    # Compute scores if requested: T = (X - mu) %*% W_eff
    if (scores != "none" && !is.null(fit$x_weights) && !is.null(fit$x_loadings)) {
      Xc <- sweep(Xr, 2L, fit$x_means, "-")
      Rmat <- crossprod(fit$x_loadings, fit$x_weights)
      Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
      W_eff <- if (!is.null(Rinv)) fit$x_weights %*% Rinv else fit$x_weights
      Tmat <- Xc %*% W_eff
      fit$scores <- if (identical(scores, "r")) Tmat else {
        # big: lift to filebacked if requested
        Tmat
      }
    } else {
      fit$scores <- NULL
    }
    .finalize_pls_fit(.post_scores(fit), "kf_pls")
  }
  
  # ---- BIGMEM BACKEND -------------------------------------------------------
  run_bigmem_simpls <- function() {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix")) stop("For backend='bigmem', y must be a big.matrix")
    
    ## Align chunk size to cache-friendly boundaries (fewer partial blocks)
    block_align <- as.integer(getOption("bigPLSR.stream.block_align", 8192L))
    if (is.na(block_align) || block_align <= 0L) block_align <- 8192L
    cs <- as.integer(chunk_size)
    if (is.na(cs) || cs <= 0L) cs <- 10000L
    chunk_aligned <- as.integer( ( (cs + block_align - 1L) %/% block_align ) * block_align )

    ## Unified: cross-products + SIMPLS (PLS1 or PLS2) with aligned blocks
    cross <- .Call(`_bigPLSR_cpp_bigmem_cross`, X@address, y@address, chunk_aligned)
    
    fit <- .Call(`_bigPLSR_cpp_simpls_from_cross`,
                 cross$XtX, cross$XtY, cross$x_means, cross$y_means,
                 as.integer(ncomp), tol)
    fit$mode <- if (ncol(cross$XtY) == 1L) "pls1" else "pls2"
    ## Ensure correct ncomp now (avoid fallback to coef dims later)
    if (is.null(fit$ncomp) || !is.finite(fit$ncomp) || fit$ncomp <= 0L) {
      if (!is.null(fit$x_weights)) fit$ncomp <- ncol(fit$x_weights)
    }
    
    # Stream scores only if requested: T = (X - mu) %*% W_eff
    
    # Before streaming T, build W_eff with a selectable solver:
    if (identical(scores, "big") || identical(scores, "r")) {
      # Prepare PLS-style weights for streaming: W_eff = W %*% solve(P'W)
      style  <- getOption("bigPLSR.scores_style", "pls")       # hidden
      solver <- match.arg(getOption("bigPLSR.simpls.solve", "chol"),
                          c("chol","tri","qr","solve"))
      W_eff  <- fit$x_weights
      if (!is.null(fit$x_weights) && !is.null(fit$x_loadings) && identical(style, "pls")) {
        Rmat <- crossprod(fit$x_loadings, fit$x_weights)       # A×A, typically SPD
        Rinv <- NULL
        if (solver %in% c("chol","tri")) {
          Rinv <- tryCatch({
            U <- chol(Rmat, pivot = FALSE)
            if (solver == "chol") chol2inv(U) else backsolve(U, diag(nrow(Rmat)))
          }, error = function(e) NULL)
        }
        if (is.null(Rinv) && solver == "qr") {
          Rinv <- tryCatch(solve(qr(Rmat), diag(nrow(Rmat))), error = function(e) NULL)
        }
        if (is.null(Rinv)) {
          Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
        }
        if (!is.null(Rinv)) {
          W_eff <- fit$x_weights %*% Rinv 
        }
      }
      local_sink <- NULL
      if (identical(scores, "big")) {
        if (is.null(scores_bm)) {
          # CRAN: never write outside tempdir() by default
          default_path <- getOption("bigPLSR.backingpath_default", tempdir())
          local_sink <- bigmemory::filebacked.big.matrix(
            nrow = nrow(X), ncol = as.integer(fit$ncomp), type = "double",
            backingfile = scores_backingfile %||% "scores.bin",
            backingpath = scores_backingpath %||% default_path,
            descriptorfile = scores_descriptorfile %||% "scores.desc"
          )
        } else if (inherits(scores_bm, "big.matrix")) {
          local_sink <- scores_bm
        } else if (inherits(scores_bm, "big.matrix.descriptor")) {
          local_sink <- bigmemory::attach.big.matrix(scores_bm)
        }
      }
      fit$scores <- .Call(`_bigPLSR_cpp_stream_scores_given_W`,
                          X@address, W_eff, fit$x_means,
                          chunk_aligned,
                          if (is.null(local_sink)) NULL else local_sink,
                          identical(scores, "big"))
    } else {
      fit$scores <- NULL
    }
    fit <- .finalize_pls_fit(.post_scores(fit), "simpls")
    if (.bigPLSR_should_store_X(X)) fit$X <- X
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
    
    # Align chunk_size to cache-friendly block size
    blk <- .bigPLSR_stream_block_size(nrow(X), chunk_size)
    
    if (mode == "pls1" && ncol(y) != 1L) stop("mode='pls1' requires y to have one column")
    
    # Align streaming chunk size to cache boundaries for better throughput
    block_align <- as.integer(getOption("bigPLSR.stream.block_align", 8192L))
    if (is.na(block_align) || block_align <= 0L) block_align <- 8192L
    cs <- as.integer(chunk_size); if (is.na(cs) || cs <= 0L) cs <- 10000L
    chunk_aligned <- as.integer(((cs + block_align - 1L) %/% block_align) * block_align)
    
    sink_bm <- NULL
    if (identical(scores, "big") && scores_target == "existing") {
      if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix")) {
        sink_bm <- scores_bm
      } else if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix.descriptor")) {
        sink_bm <- bigmemory::attach.big.matrix(scores_bm)
      } else if (!is.null(scores_backingfile)) {
        # CRAN: never write outside tempdir() by default
        default_path <- getOption("bigPLSR.backingpath_default", tempdir())
        sink_bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(X), ncol = as.integer(ncomp), type = "double",
          backingfile = scores_backingfile,
          backingpath = scores_backingpath %||% default_path,
          descriptorfile = scores_descriptorfile %||% "scores.desc"
        )
      } else {
        stop("scores_target='existing' requires scores_bm or backingfile/path/descriptorfile")
      }
    }
    
    if (mode == "pls1") {
      fit <- pls_streaming_bigmemory(
        X@address, y@address,
        as.integer(ncomp), chunk_aligned,
        center = TRUE, scale = FALSE,
        tol = tol,
        return_big = identical(scores, "big")
      )
      fit$mode <- "pls1"
    } else {
      fit <- big_plsr_stream_fit_nipals(
        X@address, y@address, as.integer(ncomp),
        center = TRUE, scale = FALSE,
        chunk_size = chunk_aligned,
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
        # CRAN: never write outside tempdir() by default
        default_path <- getOption("bigPLSR.backingpath_default", tempdir())
        bm <- bigmemory::filebacked.big.matrix(
          nrow = nrow(fit$scores), ncol = ncol(fit$scores), type = "double",
          backingfile = scores_backingfile %||% "scores.bin",
          backingpath = scores_backingpath %||% default_path,
          descriptorfile = scores_descriptorfile %||% "scores.desc"
        )
        bm[,] <- fit$scores
        fit$scores <- bm
      }
    }
    
    fit <- .finalize_pls_fit(.post_scores(fit), "nipals")
        if (.bigPLSR_should_store_X(X)) fit$X <- X
    .maybe_threshold(fit)
  }
  
  run_bigmem_kernelpls <- function(kind) {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    gram_mode <- getOption("bigPLSR.kpls_gram", "cols")
    use_rows  <- identical(gram_mode, "rows")
    
    # Align chunk_size to cache-friendly block size
    blk <- .bigPLSR_stream_block_size(nrow(X), chunk_size)
    
    if (is.null(chunk_cols)) chunk_cols_loc <- max(1024L, as.integer(0.1 * nrow(X))) else chunk_cols_loc <- as.integer(chunk_cols)
    if (identical(gram_mode, "auto")) {
      use_rows <- (nrow(X) > 4L * ncol(X))
    }
    if (use_rows) {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_xxt`,
                   X@address, y@address,
                   as.integer(ncomp), as.integer(blk), as.integer(chunk_cols_loc),
                   TRUE,
                   identical(scores, "big"))
    } else {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_cols`,
                   X@address, y@address,
                   as.integer(ncomp), as.integer(blk),
                   TRUE,
                   identical(scores, "big"))
    }
    fit$mode <- if (ncol(fit$coefficients) <= 1L) "pls1" else "pls2"
    if (identical(scores, "r") && inherits(fit$scores, "big.matrix")) {
      fit$scores <- as.matrix(fit$scores[])
    }
    fit <- .finalize_pls_fit(.post_scores(fit), "kernelpls")
        if (.bigPLSR_should_store_X(X)) fit$X <- X
    .maybe_threshold(fit)
  }
  
  run_bigmem_rkhs <- function() {
    if (!inherits(X, "big.matrix") || !inherits(y, "big.matrix"))
      stop("For backend='bigmem' and algorithm='rkhs', both X and y must be big.matrix")
    Xbm <- X
    Ybm <- y
    chunk_rows <- as.integer(getOption("bigPLSR.rkhs.chunk_rows", 8192L))
    chunk_cols <- as.integer(getOption("bigPLSR.rkhs.chunk_cols", 8192L))
    fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_bigmem`,
                 Xbm@address, Ybm@address,
                 as.integer(ncomp), as.numeric(tol),
                 as.character(kernel), as.numeric(gamma), as.integer(degree), as.numeric(coef0),
                 as.character(approx), if (is.null(approx_rank)) -1L else as.integer(approx_rank),
                 chunk_rows, chunk_cols)
    # finalize
    fit$mode <- if (ncol(y) == 1L) "pls1" else "pls2"
    if (identical(scores, "none")) {
      fit$scores <- NULL
    } else if (identical(scores, "r") && inherits(fit$scores, "big.matrix")) {
      fit$scores <- as.matrix(fit$scores[])
    }
    # Store training descriptor so predict() can stream cross-kernel
    fit$X_ref <- bigmemory::describe(Xbm)
    # Precompute & store HKH centering stats r,g for training kernel (streamed)
    fit$kstats <- bigPLSR_stream_kstats(
      Xbm,
      kernel = kernel, gamma = gamma, degree = degree, coef0 = coef0,
      chunk_rows = chunk_rows,
      chunk_cols = chunk_cols
    )
    .finalize_pls_fit(.post_scores(fit), "rkhs")
  }
  
  run_bigmem_kernelpls <- function(kind) {
    if (!inherits(X, "big.matrix"))
      stop("For backend='bigmem', X must be a big.matrix")
    if (!inherits(y, "big.matrix"))
      stop("For backend='bigmem', y must be a big.matrix")
    
    # choose streaming strategy (row-Gram XXᵗ vs. column streaming)
    n <- nrow(X)
    p <- ncol(X)
    gram_mode <- getOption("bigPLSR.kpls_gram", "auto")
    if (identical(gram_mode, "auto")) {
      # heuristic: prefer XXᵗ when n << p (row-Gram is small)
      gram_mode <- if (n * 1.2 < p) "rows" else "cols"
    }
    
    chunk_rows <- as.integer(getOption("bigPLSR.chunk_rows", 8192L))
    chunk_cols <- as.integer(getOption("bigPLSR.chunk_cols", max(64L, floor(0.1 * n))))
    
    if (identical(gram_mode, "rows")) {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_xxt`,
                   X@address, y@address,
                   as.integer(ncomp),
                   as.integer(chunk_rows),
                   as.integer(chunk_cols),
                   TRUE,                    # center
                   identical(scores, "big"))
    } else {
      fit <- .Call(`_bigPLSR_cpp_kpls_stream_cols`,
                   X@address, y@address,
                   as.integer(ncomp),
                   as.integer(chunk_cols),
                   TRUE,                    # center
                   identical(scores, "big"))
    }
    fit$mode <- if (ncol(y) == 1L) "pls1" else "pls2"
    
    # scores routing identical to other bigmem paths
    if (identical(scores, "none")) {
      fit$scores <- NULL
    } else if (identical(scores, "r") && inherits(fit$scores, "big.matrix")) {
      fit$scores <- as.matrix(fit$scores[])
    }
    if (.bigPLSR_should_store_X(X)) fit$X <- X
    .finalize_pls_fit(.post_scores(fit), kind)
  }
  
  # ---- Kernel RKHS Logistic PLS (bigmem) -----------------------
  run_bigmem_klogitpls <- function() {
    if (!inherits(X, "big.matrix")) stop("For backend='bigmem', X must be a big.matrix")
    yv <- if (is.matrix(y)) as.numeric(y[, 1]) else if (inherits(y, "big.matrix")) as.numeric(y[,1]) else as.numeric(y)
    if (is.factor(yv)) {
      lev <- levels(yv)
      if (length(lev) != 2L) stop("klogitpls requires a binary response (factor with 2 levels).", call. = FALSE)
      ybin <- as.numeric(yv == lev[2L]); classes <- lev
      cw <- normalize_class_weights(yv, class_weights)
    } else {
      ybin <- as.numeric(yv > 0.5); classes <- c("0","1")
      cw <- normalize_class_weights(factor(ybin, levels = c(0,1), labels = classes), class_weights)
    }
    # hyper-params (allow ... overrides if present, else options, else defaults)
    kernel <- kernel %||% getOption("bigPLSR.klogitpls.kernel", "rbf")
    gamma  <- gamma  %||% getOption("bigPLSR.klogitpls.gamma",  1 / ncol(Xr))
    degree <- degree %||% getOption("bigPLSR.klogitpls.degree", 3L)
    coef0  <- coef0  %||% getOption("bigPLSR.klogitpls.coef0",  1.0)
    # IRLS controls / class weights
#    cw     <- if (exists("class_weights", inherits = FALSE) && !is.null(class_weights)) as.numeric(class_weights) else numeric()
    itmax  <- if (exists("klogit_maxit", inherits = FALSE)) klogit_maxit else getOption("bigPLSR.klogitpls.maxit", 50L)
    ittol  <- if (exists("klogit_tol",   inherits = FALSE)) klogit_tol   else getOption("bigPLSR.klogitpls.tol",   1e-8)
    alt_it <- if (exists("klogit_alt",   inherits = FALSE)) klogit_alt   else getOption("bigPLSR.klogitpls.alt",   0L)
    u_rdg  <- getOption("bigPLSR.klogitpls.u_ridge", 1e-10)
    
    # ----- BIGMEM: streaming RKHS KPLS + IRLS -----
    Xbm <- X
    chunk_rows <- as.integer(getOption("bigPLSR.klogitpls.chunk_rows", 8192L))
    chunk_cols <- as.integer(getOption("bigPLSR.klogitpls.chunk_cols", 8192L))
    fit0 <- .Call(
      `_bigPLSR_cpp_kpls_rkhs_bigmem`,
      Xbm@address,
      matrix(ybin, ncol = 1L),
      as.integer(ncomp),
      as.numeric(tol),
      as.character(kernel),
      as.numeric(gamma),
      as.integer(degree),
      as.numeric(coef0),
      as.character("none"),
      as.integer(0L),
      chunk_rows,
      chunk_cols
    )
    T  <- as.matrix(fit0$scores)
    A  <- ncol(T)
    if (A < 1L) stop("klogitpls (bigmem): no component extracted")
    ir <- if (exists('cpp_irls_binomial', mode = 'function')) {
      cpp_irls_binomial(T, ybin, w_class = cw, maxit = itmax, tol = ittol)
    } else {
      ir <- .run_irls(T, ybin, cw, itmax, ittol)
    }
    last_u_basis <- fit0$u_basis %||% NULL
    if (alt_it > 0L) {
      for (kk in seq_len(alt_it)) {
        eta <- drop(cbind(1, T) %*% c(ir$b, ir$beta))
        p   <- .sigmoid(eta)
        fitk <- .Call(
          `_bigPLSR_cpp_kpls_rkhs_bigmem`,
          Xbm@address,
          matrix(p, ncol = 1L),
          as.integer(ncomp),
          as.numeric(tol),
          as.character(kernel),
          as.numeric(gamma),
          as.integer(degree),
          as.numeric(coef0),
          as.character("none"),
          as.integer(0L),
          chunk_rows,
          chunk_cols
        )
        T <- as.matrix(fitk$scores)
        last_u_basis <- fitk$u_basis %||% last_u_basis
        ir <- .run_irls(T, ybin, cw, itmax, ittol)
      }
    }
    # Build object; bigmem predict can stream cross-kernel later if needed
    ir <- .calibrate_weighted_logit(T, ybin, ir, cw, itmax, ittol)
    obj <- list(
      algorithm    = "klogitpls",
      family       = "binomial",
      ncomp        = A,
      classes      = classes,
      intercept    = ir$b,
      latent_coef  = ir$beta,
      class_weights = if (length(cw)) cw else NULL,
      u_basis      = last_u_basis,
      kernel       = kernel, gamma = gamma, degree = degree, coef0 = coef0,
      # Precomputed centering stats for HKH using the training kernel
      kstats       = fit0$kstats %||% bigPLSR_stream_kstats(
        Xbm,
        kernel = kernel, gamma = gamma, degree = degree, coef0 = coef0,
        chunk_rows = chunk_rows,
        chunk_cols = chunk_cols
      ) %||% NULL,
      # Training reference for streamed prediction
      X_ref       = bigmemory::describe(Xbm),
      scores       = T,
      mode         = "pls1"
    )
    return(.finalize_pls_fit(.post_scores(obj), "klogitpls"))
  }
  

  run_bigmem_kf_pls <- function() {
    if (!inherits(X, "big.matrix") || !inherits(y, "big.matrix"))
      stop("For backend='bigmem', both X and y must be big.matrix for kf_pls")
    fit <- cpp_kf_pls_bigmem(X@address, y@address, as.integer(ncomp),
                             as.integer(chunk_size), tol = tol,
                             lambda = kf_lambda, q_proc = kf_qproc)
    fit$mode <- if (ncol(y) > 1L) "pls2" else "pls1"
    # Stream scores if requested via existing kernel
    
    if (scores != "none" && !is.null(fit$x_weights) && !is.null(fit$x_loadings)) {
      local_sink <- NULL
      if (identical(scores, "big")) {
        if (is.null(scores_bm)) {
          # CRAN: never write outside tempdir() by default
          default_path <- getOption("bigPLSR.backingpath_default", tempdir())
          local_sink <- bigmemory::filebacked.big.matrix(
            nrow = nrow(X), ncol = as.integer(fit$ncomp), type = "double",
            backingfile = scores_backingfile %||% "scores.bin",
            backingpath = scores_backingpath %||% default_path,
            descriptorfile = scores_descriptorfile %||% "scores.desc"
          )
        } else if (inherits(scores_bm, "big.matrix")) {
          local_sink <- scores_bm
        } else if (inherits(scores_bm, "big.matrix.descriptor")) {
          local_sink <- bigmemory::attach.big.matrix(scores_bm)
        }
      }
      Rmat <- crossprod(fit$x_loadings, fit$x_weights)
      Rinv <- tryCatch(solve(Rmat), error = function(e) NULL)
      W_eff <- if (!is.null(Rinv)) fit$x_weights %*% Rinv else fit$x_weights
      # existing C++ streamer: _bigPLSR_cpp_stream_scores_given_W
      fit$scores <- .Call(`_bigPLSR_cpp_stream_scores_given_W`,
                          X@address, W_eff, fit$x_means,
                          as.integer(chunk_size),
                          if (is.null(local_sink)) NULL else local_sink, 
                          identical(scores, "big"))
    } else {
      fit$scores <- NULL
    }
    .finalize_pls_fit(.post_scores(fit), "kf_pls")
  }

  # ---- Dispatch on algorithm -------------------------------------------------
  if (backend == "arma") {
    if (algo_in == "simpls") {
      return(run_dense_simpls())
    } else if (algo_in == "nipals") {
      return(run_dense_nipals())
    } else if (algo_in == "klogitpls") {
      return(run_dense_klogitpls())
    } else if (algo_in == "kernelpls") {
      return(run_dense_kernelpls("kernel"))
    } else if (algo_in == "widekernelpls") {
      return(run_dense_kernelpls("wide"))
    } else if (algo_in == "rkhs") {
      return(run_dense_rkhs())
    } else if (algo_in == "klogitpls") {
      return(run_dense_klogitpls())
    } else if (algo_in == "kf_pls") {
      return(run_dense_kf_pls())
    } else if (algo_in == "sparse_kpls") {
      return(run_dense_sparse_kpls())
    } else if (algo_in == "rkhs_xy") {
      return(run_dense_rkhs_xy())
      # Dense Double RKHS (X and Y in RKHS)
      Xr <- as.matrix(if (is.matrix(X)) X else if (inherits(X,"big.matrix")) X[] else X)
      Yr <- if (is.matrix(y)) y else if (inherits(y,"big.matrix")) y[] else as.matrix(y)
      fit <- .Call(`_bigPLSR_cpp_kpls_rkhs_xy_dense`,
                   Xr, Yr, as.integer(ncomp), tol,
                   as.character(kernel_x), as.numeric(gamma_x), as.integer(degree_x), as.numeric(coef0_x),
                   as.character(kernel_y), as.numeric(gamma_y), as.integer(degree_y), as.numeric(coef0_y),
                   as.numeric(lambda_x), as.numeric(lambda_y))
      fit$mode <- if (NCOL(Yr) == 1L) "pls1" else "pls2"
          if (.bigPLSR_should_store_X(X)) fit$X <- Xr
      
      return(.finalize_pls_fit(fit, "rkhs_xy"))
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
    } else if (algo_in == "klogitpls") {
      return(run_bigmem_klogitpls())
    } else if (algo_in == "kf_pls") {
      return(run_bigmem_kf_pls())
    } else if (algo_in == "kernelpls") {
      return(run_bigmem_kernelpls("kernel"))
    } else if (algo_in == "widekernelpls") {
      return(run_bigmem_kernelpls("wide"))
    } else if (algo_in == "rkhs") {
      return(run_bigmem_rkhs())
    } else if (algo_in == "rkhs_xy") {
      stop("algorithm='rkhs_xy' is currently implemented for dense backend only.")
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

.bigPLSR_should_store_X <- function(X) {
  nr <- nrow(X); nc <- ncol(X)
  max_rows  <- getOption("bigPLSR.store_X_max_rows", 20000L)
  max_bytes <- getOption("bigPLSR.store_X_max_bytes", 64 * 1024^2) # ~64MB
  bytes <- as.double(nr) * as.double(nc) * 8
  isTRUE(nr <= max_rows && bytes <= max_bytes)
}

.bigPLSR_should_store_Kstats <- function(n) {
  # only compute K (n x n) means when reasonably small
  n <= getOption("bigPLSR.store_Kstats_max_n", 6000L)
}

.bigPLSR_make_kernel <- function(A, B, kernel = "rbf",
                                 gamma = 1 / pmax(1, ncol(A)),
                                 degree = 3L, coef0 = 1) {
  # accepte une fonction kernel(A,B) ou un nom de noyau
  if (is.function(kernel)) return(kernel(A, B))
  kernel <- tolower(as.character(kernel)[1L])
  A <- as.matrix(A); B <- as.matrix(B)
  if (is.null(gamma)) gamma <- 1 / ncol(A)
  switch(tolower(kernel),
         "linear" = A %*% t(B),
         "rbf"    = {
           an <- rowSums(A * A)
           bn <- rowSums(B * B)
           D2 <- outer(an, bn, "+") - 2 * (A %*% t(B))
           exp(-gamma * D2)
         },
         "gaussian" = {
           an <- rowSums(A * A)
           bn <- rowSums(B * B)
           D2 <- outer(an, bn, "+") - 2 * (A %*% t(B))
           exp(-gamma * D2)
         },
         "poly" = {
           G <- A %*% t(B)
           (gamma * G + coef0)^degree
         },
         "polynomial" = {
           G <- A %*% t(B)
           (gamma * G + coef0)^degree
         },
         "tanh" = {
           G <- A %*% t(B)
           tanh(gamma * G + coef0)
         },
         stop("Unknown kernel: ", kernel)
  )
}

.bigPLSR_try_store_kstats <- function(fit, X, kernel, gamma, degree, coef0, force = FALSE) {
  # If C++ already provided stats, keep them
  if (!is.null(fit$k_colmeans) && !is.null(fit$k_grandmean)) return(fit)
  n <- nrow(X)
  if (!force && !.bigPLSR_should_store_Kstats(n)) return(fit)
  # compute K stats once at training time
  K <- .bigPLSR_make_kernel(X, X, kernel, gamma, degree, coef0)
  fit$k_colmeans  <- colMeans(K)
  fit$k_grandmean <- mean(K)
  fit
}

#' Streamed centering statistics for RKHS kernels
#'
#' Compute the column means and grand mean of the kernel matrix \eqn{K(X, X)}
#' without materialising it in memory. The input design matrix must be stored as
#' a \code{bigmemory::big.matrix} (or descriptor), and the kernel is evaluated by
#' iterating over row/column chunks.
#'
#' @param Xbm A \code{bigmemory::big.matrix} (or descriptor) containing the
#'   training design matrix.
#' @param kernel Kernel name passed to [stats::kernel()] compatible helpers
#'   (\code{"linear"}, \code{"rbf"}, \code{"poly"}, \code{"sigmoid"}).
#' @param gamma,degree,coef0 Kernel hyper-parameters.
#' @param chunk_rows,chunk_cols Numbers of rows/columns to process per chunk.
#' @return A list with entries \code{r} (column means) and \code{g}
#'   (grand mean) of the kernel matrix.
#' @export
bigPLSR_stream_kstats <- function(Xbm,
                                  kernel, gamma, degree, coef0,
                                  chunk_rows = getOption("bigPLSR.predict.chunk_rows", 8192L),
                                  chunk_cols = getOption("bigPLSR.predict.chunk_cols", 8192L)) {
  Xbm <- if (inherits(Xbm, "big.matrix.descriptor")) bigmemory::attach.big.matrix(Xbm) else Xbm
  stopifnot(inherits(Xbm, "big.matrix"))
  n <- nrow(Xbm)
  col_sum <- numeric(n)
  total_sum <- 0
  r_seq <- seq.int(1L, n, by = chunk_rows)
  c_seq <- seq.int(1L, n, by = chunk_cols)
  for (r0 in r_seq) {
    r1 <- min(n, r0 + chunk_rows - 1L)
    Xr <- as.matrix(Xbm[r0:r1, , drop = FALSE])
    for (c0 in c_seq) {
      c1 <- min(n, c0 + chunk_cols - 1L)
      Xc <- as.matrix(Xbm[c0:c1, , drop = FALSE])
      Kblk <- .bigPLSR_make_kernel(Xr, Xc, kernel, gamma, degree, coef0)
      col_sum[c0:c1] <- col_sum[c0:c1] + colSums(Kblk)
      total_sum <- total_sum + sum(Kblk)
    }
  }
  list(r = col_sum / n, g = total_sum / (n * n))
}

.bigPLSR_stream_block_size <- function(n, want = NULL) {
  # Target cache-friendly alignment; user-overridable
  align <- as.integer(getOption("bigPLSR.stream.block_align", 8192L))
  if (!is.finite(align) || align <= 0L) align <- 8192L
  
  # If caller didn't pass a block size (or gave nonpositive), start from min(n, align)
  size <- if (is.null(want) || !is.finite(want) || want <= 0L) {
    min(as.integer(n), align)
  } else {
    as.integer(want)
  }
  if (!is.finite(size) || size < 1L) size = 1L
  
  # Round UP to next multiple of 'align', then clamp to n
  size <- as.integer(((size + align - 1L) %/% align) * align)
  if (size > n) size <- as.integer(n)
  
  size
}

## --- KF-PLS: attach state snapshot to a proper big_plsr object --------------
.bigPLSR_kf_state_as_fit <- function(snap, mode = c("pls1","pls2")) {
  mode <- match.arg(mode)
  # snap is a list returned from C++ with fields:
  #   x_mu (p), y_mu (m), W (p×k), P (p×k or NULL), Q (m×k or NULL),
  #   B (p×m), b0 (m), ncomp (k)
  k <- if (!is.null(snap$ncomp)) as.integer(snap$ncomp) else {
    if (!is.null(snap$W)) ncol(snap$W) else 0L
  }
  fit <- list(
    algorithm     = "kf_pls",
    mode          = mode,
    ncomp         = k,
    x_means       = as.numeric(snap$x_mu %||% numeric()),
    y_means       = as.numeric(snap$y_mu %||% numeric()),
    x_weights     = if (!is.null(snap$W))  as.matrix(snap$W)  else NULL,
    x_loadings    = if (!is.null(snap$P))  as.matrix(snap$P)  else NULL,
    y_loadings    = if (!is.null(snap$Q))  as.matrix(snap$Q)  else NULL,
    coefficients  = if (!is.null(snap$B))  as.matrix(snap$B)  else NULL,
    intercept     = as.numeric(snap$b0 %||% snap$y_mu %||% numeric())
  )
  class(fit) <- unique(c("big_plsr", class(fit)))
  .finalize_pls_fit(fit, "kf_pls")
}

# Internal: choose dense cross-products implementation for SIMPLS
# Options:
#   bigPLSR.simpls.cross = "dense_cxx" | "R" | "auto"
.bigPLSR_simpls_cross_mode <- function() {
  mode <- tolower(as.character(getOption("bigPLSR.simpls.cross", "auto")))
  if (mode %in% c("dense_cxx", "r")) return(mode)
  # auto: prefer C++ helper if available, otherwise R fallback
  if (exists("cpp_dense_cross", mode = "function")) "dense_cxx" else "r"
}
