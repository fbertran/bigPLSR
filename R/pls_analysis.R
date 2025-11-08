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
pls_predict_response <- function(object, newdata, ncomp = NULL) {
  predict(object, newdata = newdata, ncomp = ncomp, type = "response")
}

#' Predict latent scores from a PLS fit
#'
#' @inheritParams pls_predict_response
#' @return Matrix of component scores.
#' @export
pls_predict_scores <- function(object, newdata, ncomp = NULL) {
  predict(object, newdata = newdata, ncomp = ncomp, type = "scores")
}

#' Variable importance in projection (VIP) scores
#'
#' @param object A fitted PLS model.
#' @param comps Components used to compute the VIP scores. Defaults to all
#'   available components.
#'
#' @return A named numeric vector of VIP scores.
#' @export
pls_vip <- function(object, comps = NULL) {
  if (is.null(object$scores)) {
    stop("Scores are required to compute VIP. Refit with scores enabled.", call. = FALSE)
  }
  W <- as.matrix(object$x_weights)
  if (is.null(W)) {
    stop("x_weights are required to compute VIP", call. = FALSE)
  }
  Tmat <- as.matrix(object$scores)
  Q <- as.matrix(object$y_loadings)
  max_comp <- ncol(W)
  if (is.null(comps)) {
    comps <- seq_len(max_comp)
  }
  comps <- comps[comps >= 1L & comps <= max_comp]
  if (length(comps) == 0L) {
    stop("`comps` does not overlap with the available components", call. = FALSE)
  }
  Wc <- W[, comps, drop = FALSE]
  Tc <- Tmat[, comps, drop = FALSE]
  Qc <- Q
  if (ncol(Qc) < length(comps) && nrow(Qc) == length(comps)) {
    Qc <- t(Qc)
  }
  Qc <- Qc[, comps, drop = FALSE]
  SSY <- colSums(Tc^2) * colSums(Qc^2)
  if (all(SSY == 0)) {
    return(setNames(rep(0, nrow(Wc)), rownames(Wc)))
  }
  vip <- sqrt(ncol(Wc) * (Wc^2 %*% (SSY / sum(SSY))))
  vip <- as.numeric(vip)
  names(vip) <- rownames(Wc)
  vip
}

#' Plot Variable Importance in Projection (VIP)
#'
#' @param object A fitted PLS model.
#' @param comps Components to aggregate. Defaults to all available.
#' @param threshold Optional threshold to highlight influential variables.
#' @param palette Colour palette used for bars.
#' @param ... Additional parameters passed to [graphics::barplot()].
#' @export
plot_pls_vip <- function(object, comps = NULL, threshold = 1, palette = c("#4575b4", "#d73027"), ...) {
  vip <- pls_vip(object, comps = comps)
  pal <- rep_len(palette, length.out = 2L)
  cols <- ifelse(vip >= threshold, pal[2], pal[1])
  graphics::barplot(vip, col = cols, las = 2, ylab = "VIP", ...)
  if (!is.null(threshold) && is.finite(threshold)) {
    graphics::abline(h = threshold, col = "grey40", lty = 2)
  }
  invisible(vip)
}

#' Summaries for big_plsr objects
#'
#' @param object A fitted PLS model.
#' @param X Optional design matrix to recompute reconstruction metrics.
#' @param Y Optional response matrix/vector.
#' @param ... Unused.
#'
#' @return An object of class `summary.big_plsr`.
#' @export
summary.big_plsr <- function(object, X = NULL, Y = NULL, ...) {
  if (!inherits(object, "big_plsr")) {
    stop("summary.big_plsr() expects an object of class 'big_plsr'", call. = FALSE)
  }
  scores_var <- if (!is.null(object$scores)) {
    apply(as.matrix(object$scores), 2L, stats::var)
  } else {
    rep(NA_real_, object$ncomp)
  }
  explained <- if (all(is.na(scores_var))) {
    rep(NA_real_, length(scores_var))
  } else {
    scores_var / sum(scores_var)
  }
  vip <- tryCatch(pls_vip(object), error = function(e) rep(NA_real_, nrow(as.matrix(object$x_weights))))
  res <- list(
    call = object$call %||% NULL,
    algorithm = object$algorithm %||% NA_character_,
    mode = object$mode %||% NA_character_,
    ncomp = object$ncomp,
    intercept = object$intercept,
    coefficients = object$coefficients,
    score_variance = scores_var,
    explained_variance = explained,
    vip = vip
  )
  if (!is.null(X) && !is.null(Y)) {
    preds <- predict(object, X, ncomp = object$ncomp)
    Ymat <- if (is.vector(Y)) matrix(Y, ncol = 1L) else as.matrix(Y)
    residuals <- Ymat - preds
    res$residual_sum_squares <- sum(residuals^2)
    res$rmse <- sqrt(mean(residuals^2))
  }
  class(res) <- "summary.big_plsr"
  res
}

#' @export
print.summary.big_plsr <- function(x, ...) {
  cat("Partial least squares regression summary\n")
  if (!is.null(x$call)) {
    cat("Call:\n")
    print(x$call)
  }
  cat("Algorithm:", x$algorithm, "\n")
  cat("Mode:", x$mode, "\n")
  cat("Components:", x$ncomp, "\n")
  if (!is.null(x$rmse)) {
    cat(sprintf("RMSE: %.4f\n", x$rmse))
  }
  if (!all(is.na(x$score_variance))) {
    cat("Score variance:", paste(sprintf("%.4f", x$score_variance), collapse = ", "), "\n")
  }
  if (!all(is.na(x$explained_variance))) {
    cat("Explained variance (%):", paste(sprintf("%.1f", 100 * x$explained_variance), collapse = ", "), "\n")
  }
  if (!all(is.na(x$vip))) {
    cat("VIP (first 10):\n")
    vip <- x$vip
    if (length(vip) > 10L) {
      vip <- vip[seq_len(10L)]
    }
    print(vip)
  }
  invisible(x)
}

#' Plot individual scores
#'
#' @param object A fitted PLS model with scores.
#' @param comps Components to plot (length two).
#' @param labels Optional character vector of point labels.
#' @param ... Additional plotting parameters passed to [graphics::plot()].
#' @export
plot_pls_individuals <- function(object, comps = c(1L, 2L), labels = NULL, ...) {
  if (length(comps) != 2L) {
    stop("`comps` must contain exactly two component indices", call. = FALSE)
  }
  if (is.null(object$scores)) {
    stop("Scores are not available; refit the model with `scores` enabled", call. = FALSE)
  }
  scores <- as.matrix(object$scores)
  if (max(comps) > ncol(scores)) {
    stop("Requested components exceed the available scores", call. = FALSE)
  }
  graphics::plot(scores[, comps[1]], scores[, comps[2]], xlab = paste0("Comp ", comps[1]),
                 ylab = paste0("Comp ", comps[2]), ...)
  if (!is.null(labels)) {
    graphics::text(scores[, comps[1]], scores[, comps[2]], labels = labels, pos = 3)
  }
  invisible(NULL)
}

#' Plot variable loadings
#'
#' @param object A fitted PLS model.
#' @param comps Components to display (length two).
#' @param circle Logical; draw the correlation circle.
#' @param circle_col Colour of the correlation circle.
#' @param arrow_col Colour of the variable arrows.
#' @param arrow_scale Scaling applied to variable vectors.
#' @param ... Additional plotting parameters passed to [graphics::plot()].
#' @export
plot_pls_variables <- function(object, comps = c(1L, 2L), circle = TRUE,
                               circle_col = "grey80", arrow_col = "steelblue",
                               arrow_scale = 1, ...) {
  if (length(comps) != 2L) {
    stop("`comps` must contain exactly two component indices", call. = FALSE)
  }
  loadings <- as.matrix(object$x_loadings)
  if (is.null(loadings)) {
    stop("x_loadings are required to plot variable loadings", call. = FALSE)
  }
  if (max(comps) > ncol(loadings)) {
    stop("Requested components exceed the available loadings", call. = FALSE)
  }
  labels <- rownames(loadings)
  if (is.null(labels)) {
    labels <- paste0("V", seq_len(nrow(loadings)))
  }
  xr <- range(loadings[, comps[1]]) * 1.1 * arrow_scale
  yr <- range(loadings[, comps[2]]) * 1.1 * arrow_scale
  lim <- max(abs(c(xr, yr, 1)))
  graphics::plot(0, 0, type = "n", xlab = paste0("Comp ", comps[1]),
                 ylab = paste0("Comp ", comps[2]), xlim = c(-lim, lim), ylim = c(-lim, lim), ...)
  graphics::abline(h = 0, v = 0, col = "grey90", lty = 3)
  if (isTRUE(circle)) {
    theta <- seq(0, 2 * pi, length.out = 200L)
    graphics::lines(cos(theta), sin(theta), col = circle_col)
  }
  graphics::arrows(0, 0, arrow_scale * loadings[, comps[1]], arrow_scale * loadings[, comps[2]],
                   length = 0.08, angle = 20, col = arrow_col)
  graphics::text(arrow_scale * loadings[, comps[1]], arrow_scale * loadings[, comps[2]],
                 labels = labels, col = arrow_col, pos = 3)
  invisible(NULL)
}

#' PLS biplot
#'
#' @param object A fitted PLS model with scores and loadings.
#' @param comps Components to display.
#' @param scale_variables Scaling factor applied to variable loadings.
#' @param circle Logical; draw a correlation circle behind loadings.
#' @param circle_col Colour of the circle guide.
#' @param arrow_col Colour for loading arrows.
#' @param ... Additional arguments passed to [graphics::plot()].
#' @export
plot_pls_biplot <- function(object, comps = c(1L, 2L), scale_variables = 1,
                            circle = TRUE, circle_col = "grey85",
                            arrow_col = "firebrick", ...) {
  if (length(comps) != 2L) {
    stop("`comps` must contain exactly two components", call. = FALSE)
  }
  if (is.null(object$scores)) {
    stop("Scores are not available; refit the model with `scores` enabled", call. = FALSE)
  }
  scores <- as.matrix(object$scores)
  loadings <- as.matrix(object$x_loadings)
  if (max(comps) > ncol(scores) || max(comps) > ncol(loadings)) {
    stop("Requested components exceed the available dimensions", call. = FALSE)
  }
  labels <- rownames(loadings)
  if (is.null(labels)) {
    labels <- paste0("V", seq_len(nrow(loadings)))
  }
  xr <- range(scores[, comps[1]])
  yr <- range(scores[, comps[2]])
  lim <- max(abs(c(xr, yr))) * 1.1
  graphics::plot(scores[, comps[1]], scores[, comps[2]], xlab = paste0("Comp ", comps[1]),
                 ylab = paste0("Comp ", comps[2]), xlim = c(-lim, lim), ylim = c(-lim, lim), ...)
  graphics::abline(h = 0, v = 0, col = "grey90", lty = 3)
  if (isTRUE(circle)) {
    theta <- seq(0, 2 * pi, length.out = 200L)
    graphics::lines(scale_variables * cos(theta), scale_variables * sin(theta), col = circle_col)
  }
  graphics::arrows(0, 0, scale_variables * loadings[, comps[1]], scale_variables * loadings[, comps[2]],
                   length = 0.08, angle = 20, col = arrow_col)
  graphics::text(scale_variables * loadings[, comps[1]], scale_variables * loadings[, comps[2]],
                 labels = labels, col = arrow_col, pos = 3)
  invisible(NULL)
}

#' Compute information criteria for component selection
#'
#' @param object A fitted PLS model.
#' @param X Training design matrix.
#' @param Y Training response matrix or vector.
#' @param criteria Character vector specifying which criteria to compute.
#' @param max_comp Maximum number of components to consider.
#'
#' @return A data frame with RSS, RMSE, AIC and BIC per component.
#' @export
pls_information_criteria <- function(object, X, Y, max_comp = NULL) {
  if (is.null(max_comp)) {
    max_comp <- object$ncomp
  }
  Xmat <- as.matrix(X)
  Ymat <- if (is.vector(Y)) matrix(Y, ncol = 1L) else as.matrix(Y)
  comps <- seq_len(min(max_comp, object$ncomp))
  n <- nrow(Xmat)
  res <- lapply(comps, function(k) {
    preds <- predict(object, Xmat, ncomp = k)
    preds <- if (is.vector(preds)) matrix(preds, ncol = 1L) else as.matrix(preds)
    resid <- Ymat - preds
    rss <- sum(resid^2)
    rmse <- sqrt(mean(resid^2))
    k_params <- 1 + k * ncol(Ymat)
    aic <- n * log(rss / n) + 2 * k_params
    bic <- n * log(rss / n) + log(n) * k_params
    data.frame(ncomp = k, rss = rss, rmse = rmse, aic = aic, bic = bic)
  })
  do.call(rbind, res)
}

#' Component selection via information criteria
#'
#' @inheritParams pls_information_criteria
#' @param criteria Character vector specifying which criteria to compute.
#'
#' @return A list with the per-component table and the selected components.
#' @export
pls_select_components <- function(object, X, Y, criteria = c("aic", "bic"), max_comp = NULL) {
  table <- pls_information_criteria(object, X, Y, max_comp = max_comp)  
  best <- lapply(criteria, function(cri) {
    cri <- match.arg(cri, c("aic", "bic"))
    table$ncomp[which.min(table[[cri]])]
  })
  names(best) <- criteria
  list(table = table, best = best)
}

.metric_functions <- list(
  rmse = function(resid) sqrt(mean(resid^2)),
  mae = function(resid) mean(abs(resid)),
  r2 = function(resid, truth) {
    1 - sum(resid^2) / sum((truth - mean(truth))^2)
  }
)

.compute_metrics <- function(preds, truth, metrics) {
  resid <- truth - preds
  lapply(metrics, function(metric) {
    fn <- .metric_functions[[metric]]
    if (is.null(fn)) stop(sprintf("Unknown metric '%s'", metric), call. = FALSE)
    if (identical(metric, "r2")) {
      fn(resid, truth)
    } else {
      fn(resid)
    }
  })
}

#' K-fold or leave-one-out cross validation for PLS models
#'
#' @param X Predictor matrix.
#' @param Y Response matrix or vector.
#' @param ncomp Number of components to evaluate.
#' @param folds Number of folds (ignored when `type = "loo"`).
#' @param type Either "kfold" (default) or "loo".
#' @param algorithm Backend algorithm: "simpls", "nipals", "kernelpls" or
#'   "widekernelpls".
#' @param backend Backend passed to [pls_fit()].
#' @param metrics Metrics to compute (subset of "rmse", "mae", "r2").
#' @param seed Optional seed for reproducibility.
#' @return A list containing per-fold metrics and their summary across folds.
#' @export
pls_cross_validate <- function(X, Y, ncomp, folds = 5L, type = c("kfold", "loo"),
                               algorithm = c("simpls", "nipals", "kernelpls", 
                                             "widekernelpls"), backend = "arma",
                               metrics = c("rmse", "mae", "r2"), seed = NULL) {
  type <- match.arg(type)
  algorithm <- match.arg(algorithm)
  metrics <- unique(match.arg(metrics, choices = names(.metric_functions), several.ok = TRUE))
  Xmat <- as.matrix(X)
  Ymat <- if (is.vector(Y)) matrix(Y, ncol = 1L) else as.matrix(Y)
  n <- nrow(Xmat)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (identical(type, "loo")) {
    fold_ids <- seq_len(n)
  } else {
    fold_ids <- sample(rep(seq_len(folds), length.out = n))
  }
  mode <- if (ncol(Ymat) == 1L) "pls1" else "pls2"
  fold_results <- list()
  summary_rows <- list()
  for (k in seq_len(max(fold_ids))) {
    test_idx <- which(fold_ids == k)
    train_idx <- setdiff(seq_len(n), test_idx)
    fit <- pls_fit(Xmat[train_idx, , drop = FALSE], Ymat[train_idx, , drop = FALSE],
                   ncomp = ncomp, algorithm = algorithm, backend = backend, scores = "none", mode = mode)
    for (comp in seq_len(ncomp)) {
      preds <- predict(fit, Xmat[test_idx, , drop = FALSE], ncomp = comp)
      preds <- if (is.vector(preds)) matrix(preds, ncol = 1L) else as.matrix(preds)
      truth <- Ymat[test_idx, , drop = FALSE]
      metric_vals <- .compute_metrics(preds, truth, metrics)
      fold_results[[length(fold_results) + 1L]] <- data.frame(
        fold = k,
        ncomp = comp,
        metric = metrics,
        value = unlist(metric_vals),
        row.names = NULL
      )
    }
  }
  fold_df <- do.call(rbind, fold_results)
  summary_df <- stats::aggregate(value ~ ncomp + metric, data = fold_df, FUN = mean)
  list(details = fold_df, summary = summary_df)
}

#' Bootstrap confidence intervals for coefficients
#'
#' @param X Predictor matrix.
#' @param Y Response matrix or vector.
#' @param ncomp Number of components.
#' @param R Number of bootstrap replications.
#' @param algorithm Backend algorithm ("simpls", "nipals", "kernelpls" or
#'   "widekernelpls").
#' @param backend Backend argument passed to the fitting routine.
#' @param conf Confidence level.
#' @param seed Optional seed.
#' @return A list with bootstrap samples and confidence intervals.
#' @export
pls_bootstrap <- function(X, Y, ncomp, R = 100L, algorithm = c("simpls", "nipals", "kernelpls", "widekernelpls"),
                          backend = "arma", conf = 0.95, seed = NULL) {
  algorithm <- match.arg(algorithm)
  Xmat <- as.matrix(X)
  Ymat <- if (is.vector(Y)) matrix(Y, ncol = 1L) else as.matrix(Y)
  n <- nrow(Xmat)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  mode <- if (ncol(Ymat) == 1L) "pls1" else "pls2"
  coef_samples <- vector("list", R)
  for (b in seq_len(R)) {
    idx <- sample.int(n, n, replace = TRUE)
    fit <- pls_fit(Xmat[idx, , drop = FALSE], Ymat[idx, , drop = FALSE],
                   ncomp = ncomp, algorithm = algorithm, backend = backend, scores = "none", mode = mode)
    coef_samples[[b]] <- if (inherits(fit$coefficients, "big.matrix")) {
      fit$coefficients[,]
    } else {
      as.matrix(fit$coefficients)
    }
  }
  coef_arr <- simplify2array(coef_samples)
  alpha <- (1 - conf) / 2
  lower <- apply(coef_arr, 1:2, stats::quantile, probs = alpha)
  upper <- apply(coef_arr, 1:2, stats::quantile, probs = 1 - alpha)
  mean_coef <- apply(coef_arr, 1:2, mean)
  list(mean = mean_coef, lower = lower, upper = upper, samples = coef_samples)
}

#' Naive sparsity control by coefficient thresholding
#'
#' @param object A fitted PLS model.
#' @param threshold Values below this absolute magnitude are set to zero.
#' @return A modified copy of `object` with thresholded coefficients.
#' @export
pls_threshold <- function(object, threshold) {
  if (is.null(object$coefficients)) {
    stop("Object does not contain coefficient estimates", call. = FALSE)
  }
  coef_mat <- as.matrix(object$coefficients)
  coef_mat[abs(coef_mat) < threshold] <- 0
  object$coefficients <- coef_mat
  object
}

#' Select components from cross-validation results
#'
#' @param cv_result Result returned by [pls_cross_validate()].
#' @param metric Metric to optimise.
#' @param minimise Logical; whether the metric should be minimised.
#' @return Selected number of components.
#' @export
pls_cv_select <- function(cv_result, metric = c("rmse", "mae", "r2"), minimise = NULL) {
  metric <- match.arg(metric)
  summary_df <- cv_result$summary
  if (is.null(summary_df) || !metric %in% summary_df$metric) {
    stop(sprintf("Metric '%s' not available in cross-validation results", metric), call. = FALSE)
  }
  vals <- summary_df[summary_df$metric == metric, , drop = FALSE]
  if (is.null(minimise)) {
    minimise <- !identical(metric, "r2")
  }
  if (minimise) {
    vals$ncomp[which.min(vals$value)]
  } else {
    vals$ncomp[which.max(vals$value)]
  }
}
