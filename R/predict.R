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

.bigPLSR_make_kernel <- function(A, B, kernel = "rbf", gamma = NULL, degree = 3L, coef0 = 1) {
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
predict.big_plsr <- function(object, newdata, ncomp = NULL, type = c("response", "scores"), ...) {
  type <- match.arg(type)
  if (is.null(newdata)) {
    stop("`newdata` must be provided for prediction", call. = FALSE)
  }
  if (is.null(object$ncomp)) {
    stop("The supplied object does not contain component information", call. = FALSE)
  }
  if (is.null(ncomp)) {
    ncomp <- object$ncomp
  }
  if (!is.numeric(ncomp) || length(ncomp) != 1L || ncomp <= 0) {
    stop("`ncomp` must be a positive integer", call. = FALSE)
  }
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
  
  # ----- RKHS (single-kernel for X) -----------------------------------------
  if (algo %in% c("rkhs", "kernelpls", "widekernelpls")) {
    # Dual coefficients MAY be stored as "dual_coef" or "coefficients"
    alpha <- object$dual_coef %||% object$coefficients
    if (is.null(alpha)) stop("RKHS model: missing dual_coef/coefficients in fit.", call. = FALSE)
    alpha <- as.matrix(alpha)   # n x m
    b     <- object$intercept %||% object$y_means %||% rep(0, ncol(alpha))
    Xtr   <- object$X %||% object$Xtrain
    if (is.null(Xtr)) stop("RKHS predict: training X not stored in fit; refit with options(bigPLSR.store_X_max = TRUE).", call. = FALSE)
    if (inherits(newdata, "big.matrix"))
      stop("RKHS predict for big.matrix newdata not implemented yet in R; use C++ streaming path.", call. = FALSE)
    
    # kernel params (prefer *_x if present, then generic)
    kernel <- object$kernel_x %||% object$kernel %||% "rbf"
    gamma  <- object$gamma_x  %||% object$gamma  %||% (1 / ncol(Xtr))
    degree <- object$degree_x %||% object$degree %||% 3L
    coef0  <- object$coef0_x  %||% object$coef0  %||% 1
    
    # Cross-kernel and training centering (recompute K-stats if not stored)
    Kst   <- .bigPLSR_make_kernel(newdata, Xtr, kernel, gamma, degree, coef0)
    kx    <- .get_kstats_x(object)
    if (is.null(kx)) {
      # Dense fallback to derive training stats only if needed
      Ktr <- .bigPLSR_make_kernel(Xtr, Xtr, kernel, gamma, degree, coef0)
      kx  <- list(r = colMeans(Ktr), g = mean(Ktr))
    }
    m_test <- rowMeans(Kst)               # 1_n^T K_st / N_train
    Kc     <- sweep(Kst, 2, kx$r, "-")    # subtract train col-means
    Kc     <- sweep(Kc,   1, m_test,  "-")# subtract test row-means
    Kc     <- Kc + kx$g                   # add train grand-mean

    Yhatc <- Kc %*% alpha                         # n_test x m
    Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
    colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
    return(Yhat)
  }
  
  # ----- Double RKHS (X and Y in RKHS) --------------------------------------
  if (algo %in% c("rkhs_xy")) {
    alpha <- object$dual_coef %||% object$coefficients
    if (is.null(alpha)) stop("RKHS-XY model: missing dual_coef/coefficients in fit.", call. = FALSE)
    alpha <- as.matrix(alpha)
    b     <- object$intercept %||% object$y_means %||% rep(0, ncol(alpha))
    Xtr   <- object$X %||% object$Xtrain
    if (is.null(Xtr)) stop("RKHS-XY predict: training X not stored in fit; refit with options(bigPLSR.store_X_max = TRUE).", call. = FALSE)
    if (inherits(newdata, "big.matrix"))
      stop("RKHS-XY predict for big.matrix newdata not implemented yet in R; use C++ streaming path.", call. = FALSE)
    
    # X-kernel parameters (Y-kernel not needed at prediction time)
    kernel <- object$kernel_x %||% object$kernel %||% "rbf"
    gamma  <- object$gamma_x  %||% object$gamma  %||% (1 / ncol(Xtr))
    degree <- object$degree_x %||% object$degree %||% 3L
    coef0  <- object$coef0_x  %||% object$coef0  %||% 1
    
    Kst   <- .bigPLSR_make_kernel(newdata, Xtr, kernel, gamma, degree, coef0)
    kx    <- .get_kstats_x(object)
    if (is.null(kx))
      stop("RKHS-XY predict: missing training kernel centering stats; refit with bigPLSR >= version storing kstats_x.", call. = FALSE)
    Kc    <- .bigPLSR_center_cross_kernel(Kst, r_train = kx$r, g_train = kx$g)
    
    Yhatc <- Kc %*% alpha
    Yhat  <- sweep(Yhatc, 2, as.numeric(b), FUN = "+")
    colnames(Yhat) <- colnames(object$coefficients) %||% colnames(object$dual_coef) %||% colnames(object$y_means)
    return(Yhat)
  }
  
  # ----- Kernel logistic PLS (classification) -------------------------------
  if (algo %in% c("klogitpls")) {
    if (is.null(object$u_basis))
      stop("klogitpls predict: missing u_basis; refit with bigPLSR >= this patch.", call. = FALSE)
    U  <- as.matrix(object$u_basis)  # n_train x A
    Xtr <- object$Xtrain %||% object$X
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
  
  # ----- Fallback: existing linear/SIMPLS/NIPALS predict logic --------------
  comps <- seq_len(min(ncomp, object$ncomp))
  proj <- .pls_projection(object, comps)
  Xc <- .pls_center_newdata(newdata, object$x_means %||% object$x_center)
  scores <- Xc %*% proj$W %*% proj$M
  colnames(scores) <- paste0("t", proj$comps)
  if (identical(type, "scores")) {
    return(scores)
  }
  coef_mat <- proj$W %*% proj$M %*% t(proj$Q)
  intercept <- object$intercept
  if (is.null(intercept) && !is.null(object$y_means) && !is.null(object$x_means)) {
    intercept <- drop(object$y_means - object$x_means %*% coef_mat)
  }
  preds <- Xc %*% coef_mat
  if (!is.null(intercept)) {
    preds <- sweep(preds, 2L, intercept, FUN = "+")
  }
  if (ncol(preds) == 1L) {
    return(drop(preds))
  }
  preds
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
