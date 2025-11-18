#' Internal: resolve training reference for RKHS predictions
#'
#' Accepts:
#' - dense matrix (returned as-is)
#' - big.matrix (returned as-is)
#' - big.matrix.descriptor (attached and returned)
#'
#' Sources (priority):
#'   object$X, object$Xtrain, ...$Xtrain, ...$X_ref, object$X_ref
#'
#' @keywords internal
.resolve_training_ref <- function(obj, dots) {
  Xref <- obj$X %||% obj$Xtrain %||% dots$Xtrain %||% dots$X_ref %||% obj$X_ref
  if (inherits(Xref, "big.matrix.descriptor")) {
    return(bigmemory::attach.big.matrix(Xref))
  }
  # Allow passing a filebacked big.matrix that user attached themselves,
  # or a plain dense matrix.
  Xref
}

# Internal helper to extract projection matrices for a subset of components
.pls_projection <- function(object, comps) {
  if (is.null(object$x_weights) || is.null(object$x_loadings) || is.null(object$y_loadings)) {
    stop("Object must contain x_weights, x_loadings and y_loadings to compute projections", call. = FALSE)
  }
  W <- as.matrix(object$x_weights)
  P <- as.matrix(object$x_loadings)
  Q <- as.matrix(object$y_loadings)
  max_comp <- ncol(W)
  comps <- sort(unique(comps))
  comps <- comps[comps >= 1L & comps <= max_comp]
  if (length(comps) == 0L) {
    stop("Requested components are not available in the fitted object", call. = FALSE)
  }
  if (ncol(P) < max_comp) {
    P <- t(P)
  }
  if (ncol(Q) < max_comp && nrow(Q) == max_comp) {
    Q <- t(Q)
  }
  Wc <- W[, comps, drop = FALSE]
  Pc <- P[, comps, drop = FALSE]
  Qc <- Q[, comps, drop = FALSE]
  M <- tryCatch(solve(t(Pc) %*% Wc), error = function(e) {
    stop("Failed to invert t(P) %*% W for the selected components", call. = FALSE)
  })
  list(W = Wc, P = Pc, Q = Qc, M = M, comps = comps)
}

.pls_center_newdata <- function(newdata, means) {
  if (is.null(means)) {
    return(as.matrix(newdata))
  }
  X <- as.matrix(newdata)
  if (length(means) != ncol(X)) {
    stop("Length of x-means does not match the number of columns in newdata", call. = FALSE)
  }
  sweep(X, 2L, means, FUN = "-")
}

# Center cross-kernel using training statistics.
# K_*c = K_* - 1_m r^T - c_* 1_n^T + g
.bigPLSR_center_cross_kernel <- function(K_star, r_train, g_train) {
  c_star <- rowMeans(K_star)
  K_star - outer(rep(1, nrow(K_star)), r_train) - outer(c_star, rep(1, ncol(K_star))) + g_train
}

.bigPLSR_get_train_kstats <- function(object, kernel, gamma, degree, coef0) {
  # Prefer saved stats if present
  r <- object$k_colmeans %||% NULL
  g <- object$k_grandmean %||% NULL
  if (!is.null(r) && !is.null(g)) return(list(r = as.numeric(r), g = as.numeric(g)))
  # Else recompute from training X (dense fallback)
  Xtr <- object$X %||% object$Xtrain
  if (is.null(Xtr)) stop("RKHS predict: training X not stored in fit; refit with options(bigPLSR.store_X_max = TRUE).", call. = FALSE)
  K <- .bigPLSR_make_kernel(Xtr, Xtr, kernel = kernel, gamma = gamma, degree = degree, coef0 = coef0)
  list(r = colMeans(K), g = mean(K))
}


#' Predict method for big_plsr objects
#'
#' @param object A fitted PLS model produced by [pls_fit()].
#' @param newdata Matrix or `bigmemory::big.matrix` with predictor values.
#' @param ncomp Number of components to use for prediction.
#' @param type Either "response" (default) or "scores".
#' @param ... Unused, for compatibility with the generic.
#' 
#' @return Predicted responses or component scores.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' predict(fit, X, ncomp = 2)
predict.big_plsr <- function(object, newdata, ncomp = NULL,
                             type = c("response", "scores", "prob", "class"),
                             ...) {
  type <- match.arg(type)
  dots <- list(...)
  if (is.null(newdata)) {
    stop("`newdata` must be provided for prediction", call. = FALSE)
  }
  
  ## grab extra args early (may include Xtrain for bigmem)
  dots <- list(...)
  
  ## tiny helpers (local to predict to avoid NAMESPACE clutter)
  if (is.null(object$ncomp)) {
    stop("The supplied object does not contain component information", call. = FALSE)
  }
  if (is.null(ncomp)) {
    ncomp <- object$ncomp
  }
  if (!is.numeric(ncomp) || length(ncomp) != 1L || ncomp <= 0) {
    stop("`ncomp` must be a positive integer", call. = FALSE)
  }
  
  .attach_if_desc <- function(x) {
    if (inherits(x, "big.matrix")) return(x)
    if (inherits(x, "big.matrix.descriptor")) return(bigmemory::attach.big.matrix(x))
    x
  }
  
  .as_matrix_if_bm <- function(x) {
    if (inherits(x, "big.matrix")) return(as.matrix(x[,, drop = FALSE]))
    if (inherits(x, "big.matrix.descriptor")) return(as.matrix(bigmemory::attach.big.matrix(x)[,, drop = FALSE]))
    as.matrix(x)
  }
  
  
  ## ---- streamed centering helpers for kernels (training set) ----------------
  ## Compute r = colMeans(K(X,X)) and g = mean(K(X,X)) by streaming blocks.
  .stream_kstats <- function(Xtrain_bm, kernel, gamma, degree, coef0,
                             chunk_rows = getOption("bigPLSR.predict.chunk_rows", 
                                                    .bigPLSR_stream_block_size(nrow(newdata), NA_integer_)),
                             chunk_cols = getOption("bigPLSR.predict.chunk_cols", 
                                                    .bigPLSR_stream_block_size(nrow(newdata), NA_integer_))) {
    Xbm <- .attach_if_desc(Xtrain_bm)
    stopifnot(inherits(Xbm, "big.matrix"))
    n <- nrow(Xbm); p <- ncol(Xbm)
    col_sum <- numeric(n); total_sum <- 0
    ## nested row/col loops: accumulate column sums and total
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
  
  ## Stream cross-kernel centering Kc(new,train)·V without materialising Kc
  ##  - If V is NULL → return row_means (for diagnostics) or an empty matrix
  ##  - For regression: V = alpha (n_train x m) and returns Kc %*% alpha
  ##  - For klogitpls:  V = u_basis (n_train x A) and returns Tnew = Kc %*% u_basis
  .stream_cross_apply <- function(newdata, Xtrain_ref, V,
                                  kernel, gamma, degree, coef0,
                                  r_train, g_train,
                                  chunk_cols = getOption("bigPLSR.predict.chunk_cols", 8192L)) {
    Xt <- .attach_if_desc(Xtrain_ref)
    ntr <- if (inherits(Xt, "big.matrix")) nrow(Xt) else nrow(Xt)
    Xnew <- if (inherits(newdata, "big.matrix") || inherits(newdata, "big.matrix.descriptor"))
      .as_matrix_if_bm(newdata) else as.matrix(newdata)
    nnew <- nrow(Xnew)
    m_out <- if (is.null(V)) 0L else ncol(V)
    out <- if (m_out > 0L) matrix(0, nnew, m_out) else NULL
    row_sum <- numeric(nnew)
    ## First pass: accumulate row sums of K(new, train) to get row means
    c_seq <- seq.int(1L, ntr, by = chunk_cols)
    for (c0 in c_seq) {
      c1 <- min(ntr, c0 + chunk_cols - 1L)
      Xc <- if (inherits(Xt, "big.matrix")) as.matrix(Xt[c0:c1, , drop = FALSE]) else Xt[c0:c1, , drop = FALSE]
      Kblk <- .bigPLSR_make_kernel(Xnew, Xc, kernel, gamma, degree, coef0) # (nnew x nc)
      row_sum <- row_sum + rowSums(Kblk)
    }
    row_mean <- row_sum / ntr
    ## Second pass: apply centered block to V and accumulate
    if (m_out > 0L) {
      for (c0 in c_seq) {
        c1 <- min(ntr, c0 + chunk_cols - 1L)
        Xc <- if (inherits(Xt, "big.matrix")) as.matrix(Xt[c0:c1, , drop = FALSE]) else Xt[c0:c1, , drop = FALSE]
        Kblk <- .bigPLSR_make_kernel(Xnew, Xc, kernel, gamma, degree, coef0)  # (nnew x nc)
        ## center: Kc = Kblk - 1 r_train[c0:c1]^T - row_mean 1^T + g_train
        Kblk <- Kblk -
          matrix(1, nnew, 1) %*% t(r_train[c0:c1]) -
          row_mean %o% rep(1, ncol(Kblk)) +
          g_train
        out <- out + Kblk %*% V[c0:c1, , drop = FALSE]
      }
    }
    out
  }
  
  # optional helpers from dots
  X_ref <- .resolve_training_ref(object, dots)
  chunk_rows <- as.integer(dots$chunk_rows %||% getOption("bigPLSR.predict.chunk_rows",
                                                          .bigPLSR_stream_block_size(nrow(newdata), NA_integer_)))
  chunk_cols <- as.integer(dots$chunk_cols %||% getOption("bigPLSR.predict.chunk_cols",
                                                          .bigPLSR_stream_block_size(nrow(newdata), NA_integer_)))
  
  threshold  <- as.numeric(dots$threshold %||% 0.5)
  
  algo <- tolower(object$algorithm %||% "simpls")
  
  # Helper: get centering stats for the X-kernel (training set)
  .get_kstats_x <- function(obj) {
    # Preferred: list with r and g
    if (!is.null(obj$kstats_x) && is.list(obj$kstats_x) &&
        !is.null(obj$kstats_x$r) && !is.null(obj$kstats_x$g)) {
      return(list(r = obj$kstats_x$r, g = obj$kstats_x$g))
    }
    # X-specific fields
    if (!is.null(obj$kx_colmeans) && !is.null(obj$kx_grandmean)) {
      return(list(r = obj$kx_colmeans, g = obj$kx_grandmean))
    }
    # Generic fields (single-kernel trainers)
    if (!is.null(obj$k_colmeans) && !is.null(obj$k_grandmean)) {
      return(list(r = obj$k_colmeans, g = obj$k_grandmean))
    }
    NULL
  }
  

  
  ## Acquire kernel params consistently for RKHS-family models
  .get_kparams <- function(obj, Xtr) {
    list(
      kernel = .coerce_kernel_name(obj$kernel_x %||% obj$kernel %||% getOption("bigPLSR.kernel", "rbf")),
      gamma  = obj$gamma_x  %||% obj$gamma  %||% (1 / ncol(.as_matrix_if_bm(Xtr))),
      degree = obj$degree_x %||% obj$degree %||% 3L,
      coef0  = obj$coef0_x  %||% obj$coef0  %||% 1
    )
  }
  
  .get_klogitpls_params <- function(obj, Xtr) {
    list(
      kernel = .coerce_kernel_name(obj$kernel_x %||% obj$kernel %||% getOption("bigPLSR.klogitpls.kernel", "rbf")),
      gamma  = obj$gamma_x  %||% obj$gamma  %||% getOption("bigPLSR.klogitpls.gamma",  1 / ncol(.as_matrix_if_bm(Xtr))),
      degree = obj$degree_x %||% obj$degree %||% getOption("bigPLSR.klogitpls.degree", 3L),
      coef0  = obj$coef0_x  %||% obj$coef0  %||% getOption("bigPLSR.klogitpls.coef0",  1.0)
    )
  }

  ## Resolve training reference (dense matrix / big.matrix / descriptor / via ...)
  .resolve_training_ref <- function(obj, dots) {
    Xref <- dots$Xtrain %||% dots$X_ref %||% obj$X %||% obj$Xtrain %||% obj$X_ref
    if (inherits(Xref, "big.matrix.descriptor")) {
      return(bigmemory::attach.big.matrix(Xref))
    }
    Xref
  }

  # ---- Solver selection mirroring training-time option -----------------------
  .predict_solver <- function() {
    m <- getOption("bigPLSR.simpls.solve", "chol")
    if (!is.character(m) || length(m) != 1L) m <- "chol"
    m <- tolower(m)
    if (!m %in% c("chol","tri","qr","solve")) m <- "chol"
    m
  }
  
  # Right-solve utility: return X = W %*% solve(R) without forming solve(R)
  .predict_right_solve <- function(R, W, method = .predict_solver()) {
    k <- ncol(R)
    if (!is.matrix(R) || nrow(R) != k) stop("R must be square")
    method <- tolower(method)
    if (method %in% c("chol","tri")) {
      out <- try({
        U <- chol(R, pivot = FALSE)
        # W %*% inv(R) == W %*% inv(U) %*% inv(t(U))
        Z <- backsolve(U, t(W), transpose = TRUE)   # solve(t(U), t(W))
        t(backsolve(U, Z, transpose = FALSE))       # solve(U, t(...))
      }, silent = TRUE)
      if (!inherits(out, "try-error")) return(out)
    } else if (method == "qr") {
      out <- try({
        q <- qr(R)
        t(qr.solve(q, t(W)))
      }, silent = TRUE)
      if (!inherits(out, "try-error")) return(out)
    }
    # Fallback
    W %*% solve(R)
  }
  
  # ---- tiny helpers used below ---------------------------------------------
  .as_row_numeric <- function(x) {
    x <- as.numeric(x); dim(x) <- c(1L, length(x)); x
  }
  .coerce_kernel_name <- function(k, default = "rbf") {
    if (is.null(k)) return(default)
    if (is.character(k)) return(tolower(k[1]))
    k1 <- tryCatch(as.character(k)[1], error = function(e) NA_character_)
    if (is.na(k1)) default else tolower(k1)
  }
  .safe_centered_scores <- function(X, mu, W, M = NULL) {
    # T = (X - mu) %*% (W %*% M)  without materializing Xc
    X   <- .as_matrix_if_bm(X)
    mu  <- .as_row_numeric(mu %||% rep(0, ncol(X)))
    WM  <- if (is.null(M)) W else W %*% M
    TW0 <- X %*% WM
    shift <- drop(mu %*% WM)          # 1 x ncomp → numeric
    sweep(TW0, 2L, shift, FUN = "-")
  }

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
  
  # ----- RKHS (single-kernel for X) -----------------------------------------
  if (algo %in% c("rkhs", "kernelpls", "widekernelpls")) {
    # Dual coefficients MAY be stored as "dual_coef" or "coefficients"
    alpha <- object$dual_coef %||% object$coefficients
    if (is.null(alpha)) stop("RKHS model: missing dual_coef/coefficients in fit.", call. = FALSE)
    alpha <- as.matrix(alpha)   # n x m
    b     <- object$intercept %||% object$y_means %||% rep(0, ncol(alpha))
    Xtr   <- .resolve_training_ref(object, list(...))
    if (is.null(Xtr)) stop("RKHS predict: training X not available. Pass Xtrain=... or refit with options(bigPLSR.store_X_max=TRUE).", call. = FALSE)
    # kernel params (prefer *_x if present, then generic; robust coerce)
    kp    <- .get_kparams(object, Xtr)
    
    ## fast path: dense training in memory (unchanged)
    if (!inherits(Xtr, "big.matrix") && !inherits(Xtr, "big.matrix.descriptor")) {
      Kst   <- .bigPLSR_make_kernel(newdata, Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      kstat <- .bigPLSR_get_train_kstats(object, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      Kc    <- .bigPLSR_center_cross_kernel(Kst, r_train = kstat$r, g_train = kstat$g)
      if (identical(type, "scores")) {
        # ---- Center-free RKHS scores: T = H K(new,train) H U
        U <- object$u_basis %||% object$U %||% stop("RKHS predict(scores): missing u_basis in fit.")
        U <- as.matrix(U)
        # U0 = U - 1 * mean(U)  (column-wise mean)
        U0 <- sweep(U, 2L, colMeans(U), FUN = "-")
        V  <- Kst %*% U0
        T  <- sweep(V, 2L, colMeans(V), FUN = "-")
        colnames(T) <- paste0("t", seq_len(ncol(T)))
        return(T)
      } else {
        Yhatc <- Kc %*% alpha                         # m columns
        Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
        colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
        return(Yhat)
      }
    } else {
      ## streamed bigmem path
      kstat <- object$kstats
      if (is.null(kstat) || is.null(kstat$r) || is.null(kstat$g)) {
        kstat <- .stream_kstats(Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      }
      Yhatc <- .stream_cross_apply(newdata, Xtr, alpha, kp$kernel, kp$gamma, kp$degree, kp$coef0,
                                   r_train = kstat$r, g_train = kstat$g)
      Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
      colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
      return(Yhat)
    }
  }
  
  # ----- Double RKHS (X and Y in RKHS) --------------------------------------
  if (algo %in% c("rkhs_xy")) {
    alpha <- object$dual_coef %||% object$coefficients
    if (is.null(alpha)) stop("RKHS-XY model: missing dual_coef/coefficients in fit.", call. = FALSE)
    alpha <- as.matrix(alpha)
    b     <- object$intercept %||% object$y_means %||% rep(0, ncol(alpha))
    Xtr   <- .resolve_training_ref(object, list(...))
    if (is.null(Xtr)) stop("RKHS-XY predict: training X not available. Pass Xtrain=... or refit with options(bigPLSR.store_X_max=TRUE).", call. = FALSE)
    # kernel params (prefer *_x if present, then generic; robust coerce)
    kp    <- .get_kparams(object, Xtr)
    
    if (!inherits(Xtr, "big.matrix") && !inherits(Xtr, "big.matrix.descriptor")) {
      Kst   <- .bigPLSR_make_kernel(newdata, Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      kstat <- .bigPLSR_get_train_kstats(object, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      Kc    <- .bigPLSR_center_cross_kernel(Kst, r_train = kstat$r, g_train = kstat$g)
      if (identical(type, "scores")) {
        U <- object$u_basis %||% object$U %||% stop("RKHS-XY predict(scores): missing u_basis in fit.")
        U <- as.matrix(U)
        U0 <- sweep(U, 2L, colMeans(U), FUN = "-")
        V  <- Kst %*% U0
        T  <- sweep(V, 2L, colMeans(V), FUN = "-")
        colnames(T) <- paste0("t", seq_len(ncol(T)))
        return(T)
      } else {
        Yhatc <- Kc %*% alpha
        Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
        colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
        return(Yhat)
      }
    } else {
      kstat <- object$kstats
      if (is.null(kstat) || is.null(kstat$r) || is.null(kstat$g)) {
        kstat <- .stream_kstats(Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
      }
      Yhatc <- .stream_cross_apply(newdata, Xtr, alpha, kp$kernel, kp$gamma, kp$degree, kp$coef0,
                                   r_train = kstat$r, g_train = kstat$g)
      Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
      colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
      return(Yhat)
    }
  }
  
  # ----- Kernel Logistic PLS (klogitpls) ------------------------------------
  if (algo %in% c("klogitpls")) {
    uB <- object$u_basis
    if (is.null(uB)) stop("klogitpls predict: missing u_basis; refit or update to latest fit.", call. = FALSE)
    uB <- as.matrix(uB)  # n_train x A
    A  <- ncol(uB)
    if (is.null(ncomp)) ncomp <- A
    comps <- seq_len(min(ncomp, A))
    uB_use <- uB[, comps, drop = FALSE]
    
    Xtr   <- X_ref %||% object$X %||% object$Xtrain
    if (is.null(Xtr)) stop("klogitpls predict: training X not available. Pass Xtrain=... or refit with options(bigPLSR.store_X_max=TRUE).", call. = FALSE)
    
    # kernel params
    # X_ref may be a descriptor; attach if needed
    ## Acquire kernel params consistently for RKHS-family models
    kp    <- .get_klogitpls_params(object, Xtr)
    
    # Build K(new, train) in memory or by streaming
    makeKcross <- function(Xnew, Xtrain) {
      if (inherits(Xnew, "big.matrix") && inherits(Xtrain, "big.matrix")) {
        # stream both sides via blocks
        nstar <- nrow(Xnew); ntr <- nrow(Xtrain)
        Kst   <- matrix(0.0, nstar, ntr)
        for (j0 in seq(1L, ntr, by = chunk_cols)) {
          j1 <- min(ntr, j0 + chunk_cols - 1L)
          Xtb <- Xtrain[j0:j1, , drop = FALSE]
          if (inherits(Xnew, "big.matrix")) {
            Xnb <- Xnew[, , drop = FALSE]
            G   <- .bigPLSR_make_kernel(Xnb, Xtb, kp$kernel, kp$gamma, kp$degree, kp$coef0)
          } else {
            G   <- .bigPLSR_make_kernel(Xnew, Xtb, kp$kernel, kp$gamma, kp$degree, kp$coef0)
          }
          Kst[, j0:j1] <- G
        }
        Kst
      } else if (inherits(Xtrain, "big.matrix")) {
        nstar <- nrow(Xnew); ntr <- nrow(Xtrain)
        Kst   <- matrix(0.0, nstar, ntr)
        for (j0 in seq(1L, ntr, by = chunk_cols)) {
          j1 <- min(ntr, j0 + chunk_cols - 1L)
          Xtb <- Xtrain[j0:j1, , drop = FALSE]
          G   <- .bigPLSR_make_kernel(as.matrix(Xnew), Xtb, kp$kernel, kp$gamma, kp$degree, kp$coef0)
          Kst[, j0:j1] <- G
        }
        Kst
      } else {
        .bigPLSR_make_kernel(as.matrix(Xnew), as.matrix(Xtrain), kp$kernel, kp$gamma, kp$degree, kp$coef0)
      }
    }
    
    # training centering stats (use saved if available, else compute dense)
    kstat <- object$kstats
    if (is.null(kstat)) {
      # dense fallback for small Xtrain
      Ktr <- .bigPLSR_make_kernel(as.matrix(Xtr), as.matrix(Xtr),
                                  kp$kernel, kp$gamma, kp$degree, kp$coef0)
      kstat <- list(r = colMeans(Ktr), g = mean(Ktr), n = nrow(Ktr))
    }
    if (is.null(kstat$r) || is.null(kstat$g)) {
      stop("klogitpls predict: missing training kernel centering stats; refit with bigmem path or provide Xtrain so stats can be computed.", call. = FALSE)
    }
    ntr <- length(kstat$r)
    
    # cross-kernel and centering wrt training stats
    Kst <- makeKcross(newdata, Xtr)               # n* x n_train
    if (ncol(Kst) != ntr) stop("klogitpls predict: Xtrain reference size does not match training size.", call. = FALSE)
    rowm <- rowMeans(Kst)                         # n* x 1
    Kc   <- Kst - tcrossprod(rowm, rep(1, ntr)) - 
      matrix(rep(kstat$r, each = nrow(Kst)), nrow(Kst), ntr, byrow = FALSE) + 
      kstat$g
    
    # T* = Kc * u_basis ; then eta = b + T* %*% beta ; p = sigmoid(eta)
    Tstar <- Kc %*% uB_use
    if (type == "scores") {
      colnames(Tstar) <- paste0("t", seq_len(ncol(Tstar)))
      return(Tstar)
    }
    beta <- as.numeric(object$latent_coef[comps])
    b0   <- as.numeric(object$intercept %||% 0)
    eta  <- as.numeric(b0) + as.numeric(Tstar %*% beta)
    # numerical stability
    eta  <- pmin(pmax(eta, -30), 30)
    p    <- 1.0 / (1.0 + exp(-eta))
    
    if (type %in% c("response", "prob")) {
      return(p)
    }
    if (type == "class") {
      lev <- object$classes %||% c("0","1")
      cls <- ifelse(p >= threshold, lev[2L], lev[1L])
      return(factor(cls, levels = lev))
    }
    stop("klogitpls predict: unsupported type = ", sQuote(type),
         ". Use type = 'response'/'prob', 'class', or 'scores'.", call. = FALSE)
  }
  
  # ----- Kernel logistic PLS (classification) -------------------------------
  if (algo %in% c("klogitplsOO")) {
    if (is.null(object$u_basis))
      stop("klogitpls predict: missing u_basis; refit with bigPLSR >= this patch.", call. = FALSE)
    U  <- as.matrix(object$u_basis)  # n_train x A
    
    Xtr   <- X_ref %||% object$X %||% object$Xtrain
    if (is.null(Xtr))
      stop("klogitpls predict: training X not stored (bigmem). Add a streaming predict path if needed.", call. = FALSE)
    if (inherits(newdata, "big.matrix"))
      stop("klogitpls: big.matrix newdata predict not wired yet in R.", call. = FALSE)
    kernel <- object$kernel_x %||% "rbf"
    gamma  <- object$gamma_x  %||% (1 / ncol(Xtr))
    degree <- object$degree_x %||% 3L
    coef0  <- object$coef0_x  %||% 1
    Kst    <- .bigPLSR_make_kernel(newdata, Xtr, kernel, gamma, degree, coef0)
    # center cross-kernel with *training* stats
    kx <- .get_kstats_x(object)
    if (is.null(kx)) {
      # derive from stored X if missing (dense only)
      Ktr <- .bigPLSR_make_kernel(Xtr, Xtr, kernel, gamma, degree, coef0)
      kx  <- list(r = colMeans(Ktr), g = mean(Ktr))
    }
    Kc    <- .bigPLSR_center_cross_kernel(Kst, r_train = kx$r, g_train = kx$g)
    Tnew  <- Kc %*% U                   # scores on new data
    eta   <- drop(object$intercept + as.matrix(Tnew) %*% as.numeric(object$latent_coef))
    p     <- 1/(1+exp(-pmin(eta, 35)))
    out   <- cbind(prob_0 = 1 - p, prob_1 = p)
    colnames(out) <- paste0("Pr(", object$classes, ")")
    return(out)
  }
  
  # ----- KF-PLS (linear; reuse SIMPLS-style projection safely) ---------------
  if (identical(algo, "kf_pls")) {
    # Use factors in the object (W, P, Q) if available; otherwise fall back
    W <- object$x_weights
    P <- object$x_loadings
    Q <- object$y_loadings
    if (is.null(W) || is.null(P) || is.null(Q)) {
      # Fallback to generic path below
    } else {
      # Compute W_eff = W %*% solve(P' W) with the selected solver
      solver <- .simpls_solver()
      Rmat   <- crossprod(P, W)                  # k x k
      W_eff  <- .simpls_right_solve(Rmat, W, method = solver)
      # Scores in a numerically safe way (no giant Xc)
      Xmu <- object$x_means %||% object$x_center %||% rep(0, nrow = 1L, ncol = nrow(W))
      T   <- .safe_centered_scores(newdata, Xmu, W_eff, NULL)
      if (identical(type, "scores")) {
        colnames(T) <- paste0("t", seq_len(ncol(T)))
        return(T)
      }
      # Yhat = T %*% t(Q) + intercept
      preds <- T %*% t(Q)
      b <- object$intercept %||% object$y_means
      if (!is.null(b)) preds <- sweep(preds, 2L, as.numeric(b), FUN = "+")
      if (ncol(preds) == 1L) return(drop(preds)) else return(preds)
    }
    # else: no factors → fall through to the generic linear path
  }
  
  # ----- Fallback: existing linear/SIMPLS/NIPALS predict logic --------------
  comps <- seq_len(min(ncomp, object$ncomp))
  
  # Helper: robust projection builder. Returns list(W, M, Q, comps).
  .robust_pls_projection <- function(obj, comps) {
    W <- as.matrix((obj$x_weights %||% obj$W))[, comps, drop = FALSE]
    P <- if (!is.null(obj$x_loadings)) as.matrix(obj$x_loadings)[, comps, drop = FALSE] else NULL
    Q <- if (!is.null(obj$y_loadings)) as.matrix(obj$y_loadings)[, comps, drop = FALSE] else {
      # fall back to reconstruct from coefficients if available: B = W M Q'
      if (!is.null(obj$coefficients) && ncol(as.matrix(obj$coefficients)) > 0L) {
        # If we only need response predictions, we can compute coef_mat later.
        NULL
      } else NULL
    }
    # Default M = I_k; if P present, use R = P'W.
    k <- ncol(W)
    M <- diag(1.0, nrow = k, ncol = k)
    if (!is.null(P)) {
      R <- crossprod(P, W)                 # k x k
      M <- tryCatch(solve(R),
                    error = function(e) {
                      # mild Tikhonov fallback if nearly singular
                      lam <- getOption("bigPLSR.proj.ridge", 1e-10)
                      solve(R + diag(lam, nrow(R)))
                    })
    }
    list(W = W, M = M, Q = Q, comps = comps)
  }
  
  # Center newdata robustly against obj$x_means or x_center.
  .robust_center <- function(Xnew, mu) {
    X <- if (inherits(Xnew, "big.matrix")) as.matrix(Xnew[]) else as.matrix(Xnew)
    if (is.null(mu)) return(X)  # nothing to subtract
    mu <- as.numeric(mu)
    if (length(mu) == ncol(X)) {
      X <- sweep(X, 2L, mu, FUN = "-")
    } else if (length(mu) == 1L) {
      X <- X - as.numeric(mu)
    } else {
      stop(sprintf("Centering vector length (%d) does not match ncol(newdata) (%d).",
                   length(mu), ncol(X)), call. = FALSE)
    }
    X
  }
  
  proj  <- .robust_pls_projection(object, comps)
  Xc    <- .robust_center(newdata, object$x_means %||% object$x_center)
  WM    <- proj$W %*% proj$M                    # expected p x k
  
  # If shapes don't match (e.g. KF-PLS missing P, or W/means dimensions), fall back.
  if (ncol(Xc) != nrow(WM)) {
    # Try direct coefficient prediction if available
    B <- object$coefficients
    if (!is.null(B)) {
      B <- as.matrix(B)
      if (nrow(B) != ncol(Xc)) {
        stop(sprintf("Predict shape mismatch: ncol(newdata)=%d, nrow(coefficients)=%d.",
                     ncol(Xc), nrow(B)), call. = FALSE)
      }
      preds <- Xc %*% B
      b0    <- object$intercept %||% (object$y_means %||% rep(0, ncol(preds)))
      preds <- sweep(preds, 2L, as.numeric(b0), FUN = "+")
      return(if (identical(type, "scores")) matrix(NA_real_, nrow = nrow(Xc), ncol = ncol(WM)) else preds)
    }
    stop(sprintf("Predict projection mismatch: X has %d cols but W%%*%%M has %d rows.",
                 ncol(Xc), nrow(WM)), call. = FALSE)
  }
  
  # Compute scores; if M=I this is the "raw" PLS scores.
  scores <- Xc %*% WM
  colnames(scores) <- paste0("t", proj$comps)
  if (identical(type, "scores")) return(scores)
  
  # Response prediction via coef reconstruction if Q is available, else via stored B.
  if (!is.null(proj$Q)) {
    coef_mat <- proj$W %*% proj$M %*% t(proj$Q)  # p x m
  } else if (!is.null(object$coefficients)) {
    coef_mat <- as.matrix(object$coefficients)
  } else {
    # last resort: regress Y on scores if Y-loadings missing (shouldn't happen post-fit)
    coef_mat <- qr.coef(qr(scores), .robust_center(object$Ytrain %||% matrix(0, nrow(Xc), 1), object$y_means))
    if (is.null(dim(coef_mat))) coef_mat <- matrix(coef_mat, ncol = 1L)
    # lift back to p-space: B ≈ W M Q' is not available → use WM %*% coef_scores' if needed
    coef_mat <- WM %*% coef_mat
  }
  if (nrow(coef_mat) != ncol(Xc)) {
    stop(sprintf("Reconstructed coefficients have %d rows but newdata has %d columns.",
                 nrow(coef_mat), ncol(Xc)), call. = FALSE)
  }
  preds <- Xc %*% coef_mat
  b0    <- object$intercept %||% (object$y_means %||% rep(0, ncol(preds)))
  preds <- sweep(preds, 2L, as.numeric(b0), FUN = "+")
  if (ncol(preds) == 1L) return(drop(preds))
  preds
}

## ---- Logistic head for klogitpls (RKHS → scores) ---------------------------
predict_klogitpls <- function(object, newdata, type = c("response", "scores"), ...) {
  type <- match.arg(type)
  if (is.null(object$latent_coef) || is.null(object$intercept))
    stop("klogitpls predict: missing latent_coef/intercept in fit.")
  if (is.null(object$u_basis))
    stop("klogitpls predict: missing u_basis; refit with bigPLSR >= this patch.")
  
  dots <- list(...)
  
  .attach_if_desc <- function(x) {
    if (inherits(x, "big.matrix")) return(x)
    if (inherits(x, "big.matrix.descriptor")) return(bigmemory::attach.big.matrix(x))
    x
  }
  
  .as_matrix_if_bm <- function(x) {
    if (inherits(x, "big.matrix")) return(as.matrix(x[,, drop = FALSE]))
    if (inherits(x, "big.matrix.descriptor")) return(as.matrix(bigmemory::attach.big.matrix(x)[,, drop = FALSE]))
    as.matrix(x)
  }
  
  
  ## ---- streamed centering helpers for kernels (training set) ----------------
  ## Compute r = colMeans(K(X,X)) and g = mean(K(X,X)) by streaming blocks.
  .stream_kstats <- function(Xtrain_bm, kernel, gamma, degree, coef0,
                             chunk_rows = getOption("bigPLSR.predict.chunk_rows", 
                                                    .bigPLSR_stream_block_size(nrow(newdata), NA_integer_)),
                             chunk_cols = getOption("bigPLSR.predict.chunk_cols", 
                                                    .bigPLSR_stream_block_size(nrow(newdata), NA_integer_))) {
    Xbm <- .attach_if_desc(Xtrain_bm)
    stopifnot(inherits(Xbm, "big.matrix"))
    n <- nrow(Xbm); p <- ncol(Xbm)
    col_sum <- numeric(n); total_sum <- 0
    ## nested row/col loops: accumulate column sums and total
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
  
  ## Stream cross-kernel centering Kc(new,train)·V without materialising Kc
  ##  - If V is NULL → return row_means (for diagnostics) or an empty matrix
  ##  - For regression: V = alpha (n_train x m) and returns Kc %*% alpha
  ##  - For klogitpls:  V = u_basis (n_train x A) and returns Tnew = Kc %*% u_basis
  .stream_cross_apply <- function(newdata, Xtrain_ref, V,
                                  kernel, gamma, degree, coef0,
                                  r_train, g_train,
                                  chunk_cols = getOption("bigPLSR.predict.chunk_cols", 8192L)) {
    Xt <- .attach_if_desc(Xtrain_ref)
    ntr <- if (inherits(Xt, "big.matrix")) nrow(Xt) else nrow(Xt)
    Xnew <- if (inherits(newdata, "big.matrix") || inherits(newdata, "big.matrix.descriptor"))
      .as_matrix_if_bm(newdata) else as.matrix(newdata)
    nnew <- nrow(Xnew)
    m_out <- if (is.null(V)) 0L else ncol(V)
    out <- if (m_out > 0L) matrix(0, nnew, m_out) else NULL
    row_sum <- numeric(nnew)
    ## First pass: accumulate row sums of K(new, train) to get row means
    c_seq <- seq.int(1L, ntr, by = chunk_cols)
    for (c0 in c_seq) {
      c1 <- min(ntr, c0 + chunk_cols - 1L)
      Xc <- if (inherits(Xt, "big.matrix")) as.matrix(Xt[c0:c1, , drop = FALSE]) else Xt[c0:c1, , drop = FALSE]
      Kblk <- .bigPLSR_make_kernel(Xnew, Xc, kernel, gamma, degree, coef0) # (nnew x nc)
      row_sum <- row_sum + rowSums(Kblk)
    }
    row_mean <- row_sum / ntr
    ## Second pass: apply centered block to V and accumulate
    if (m_out > 0L) {
      for (c0 in c_seq) {
        c1 <- min(ntr, c0 + chunk_cols - 1L)
        Xc <- if (inherits(Xt, "big.matrix")) as.matrix(Xt[c0:c1, , drop = FALSE]) else Xt[c0:c1, , drop = FALSE]
        Kblk <- .bigPLSR_make_kernel(Xnew, Xc, kernel, gamma, degree, coef0)  # (nnew x nc)
        ## center: Kc = Kblk - 1 r_train[c0:c1]^T - row_mean 1^T + g_train
        Kblk <- Kblk -
          matrix(1, nnew, 1) %*% t(r_train[c0:c1]) -
          row_mean %o% rep(1, ncol(Kblk)) +
          g_train
        out <- out + Kblk %*% V[c0:c1, , drop = FALSE]
      }
    }
    out
  }
  
  X_ref <- .resolve_training_ref(object, dots)
  Xtr   <- X_ref %||% object$X %||% object$Xtrain
  
  if (is.null(Xtr))
    stop("klogitpls predict: training X not available. Pass Xtrain=... or refit with options(bigPLSR.store_X_max=TRUE).")
  
  ## kernel params + centering stats
  kp    <- list(
    kernel = object$kernel_x %||% object$kernel %||% getOption("bigPLSR.klogitpls.kernel", "rbf"),
    gamma  = object$gamma_x  %||% object$gamma  %||% (1 / ncol(.as_matrix_if_bm(Xtr))),
    degree = object$degree_x %||% object$degree %||% 3L,
    coef0  = object$coef0_x  %||% object$coef0  %||% 1
  )
  kstat <- object$kstats
  if ((inherits(Xtr, "big.matrix") || inherits(Xtr, "big.matrix.descriptor")) &&
      (is.null(kstat) || is.null(kstat$r) || is.null(kstat$g))) {
    kstat <- .stream_kstats(Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
  } else if (is.null(kstat)) {
    ## try dense fallback from helpers if available
    kstat <- try(.bigPLSR_get_train_kstats(object, kp$kernel, kp$gamma, kp$degree, kp$coef0), silent = TRUE)
    if (inherits(kstat, "try-error")) stop("klogitpls predict: no centering stats for kernel.")
  }
  
  ## T_new = Kc(new, Xtrain) %*% u_basis   (stream if bigmem)
  if (inherits(Xtr, "big.matrix") || inherits(Xtr, "big.matrix.descriptor")) {
    Tnew <- .stream_cross_apply(newdata, Xtr, as.matrix(object$u_basis),
                                kp$kernel, kp$gamma, kp$degree, kp$coef0,
                                r_train = kstat$r, g_train = kstat$g)
  } else {
    Kst  <- .bigPLSR_make_kernel(newdata, Xtr, kp$kernel, kp$gamma, kp$degree, kp$coef0)
    Kc   <- .bigPLSR_center_cross_kernel(Kst, r_train = kstat$r, g_train = kstat$g)
    Tnew <- Kc %*% as.matrix(object$u_basis)
  }
  
  if (identical(type, "scores")) {
    colnames(Tnew) <- paste0("t", seq_len(ncol(Tnew)))
    return(Tnew)
  }
  eta <- drop(cbind(1, Tnew) %*% c(object$intercept, object$latent_coef))
  p   <- 1 / (1 + exp(-eta))
  as.numeric(p)
}

#' Predict responses from a PLS fit
#'
#' @param object A fitted PLS model.
#' @param newdata Predictor matrix for scoring.
#' @param ncomp Number of components to use.
#' @return A numeric matrix or vector of predictions.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' pls_predict_response(fit, X, ncomp = 2)
pls_predict_response <- function(object, newdata, ncomp = NULL) {
  predict(object, newdata = newdata, ncomp = ncomp, type = "response")
}

#' Predict latent scores from a PLS fit
#'
#' @inheritParams pls_predict_response
#' @return Matrix of component scores.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' pls_predict_scores(fit, X, ncomp = 2)
pls_predict_scores <- function(object, newdata, ncomp = NULL) {
  predict(object, newdata = newdata, ncomp = ncomp, type = "scores")
}

