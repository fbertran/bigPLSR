#' Variable importance in projection (VIP) scores
#'
#' @param object A fitted PLS model.
#' @param comps Components used to compute the VIP scores. Defaults to all
#'   available components.
#'
#' @return A named numeric vector of VIP scores.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' pls_vip(fit)
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
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' plot_pls_vip(fit)
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


#' Summarize a `big_plsr` model
#'
#' @param object A fitted PLS model.
#' @param X Optional design matrix to recompute reconstruction metrics.
#' @param Y Optional response matrix/vector.
#' @param ... Unused.
#' 
#' @return An object of class `summary.big_plsr`.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' summary(fit)
summary.big_plsr <- function(object, ..., X = NULL, Y = NULL) {
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

#' Print a `summary.big_plsr` object
#'
#' @param x A `summary.big_plsr` object.
#' @param ... Passed to lower-level print methods.
#' @return `x`, invisibly.
#' @export
#' @method print summary.big_plsr
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' print(summary(fit))
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
#' @param groups Optional factor or character vector defining groups for
#'   individuals. When supplied, group-specific colours are used and, if
#'   `ellipse = TRUE`, confidence ellipses are drawn for each group.
#' @param ellipse Logical; draw group confidence ellipses when `groups` are
#'   provided.
#' @param ellipse_level Confidence level for the ellipses (between 0 and 1).
#' @param ellipse_n Number of points used to draw each ellipse.
#' @param group_col Optional vector of colours for the groups. Recycled as
#'   needed.
#' @param ... Additional plotting parameters passed to [graphics::plot()].
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' plot_pls_individuals(fit)
plot_pls_individuals <- function(object, comps = c(1L, 2L), labels = NULL,
                                 groups = NULL, ellipse = TRUE,
                                 ellipse_level = 0.95, ellipse_n = 200L,
                                 group_col = NULL, ...) {
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
  xi <- scores[, comps[1]]
  yi <- scores[, comps[2]]
  if (!is.null(groups)) {
    groups <- as.factor(groups)
    if (length(groups) != length(xi)) {
      stop("Length of `groups` must match the number of observations", call. = FALSE)
    }
    lvl <- levels(groups)
    if (is.null(group_col)) {
      group_col <- grDevices::hcl.colors(length(lvl), palette = "Dark2")
    }
    if (length(group_col) < length(lvl)) {
      group_col <- rep_len(group_col, length(lvl))
    }
    graphics::plot(xi, yi, xlab = paste0("Comp ", comps[1]),
                   ylab = paste0("Comp ", comps[2]),
                   col = group_col[groups], pch = 19, ...)
    if (!is.null(labels)) {
      graphics::text(xi, yi, labels = labels, pos = 3, col = group_col[groups])
    }
    if (isTRUE(ellipse) && ellipse_level > 0 && ellipse_level < 1) {
      draw_ellipse <- function(x, y, level, npt, col) {
        if (length(x) < 3L) return()
        Sigma <- stats::cov(cbind(x, y))
        if (any(!is.finite(Sigma))) return()
        eig <- tryCatch(base::eigen(Sigma, symmetric = TRUE), error = function(e) NULL)
        if (is.null(eig)) return()
        vals <- pmax(eig$values, 0)
        if (all(vals == 0)) return()
        radii <- sqrt(stats::qchisq(level, df = 2) * vals)
        angle <- seq(0, 2 * pi, length.out = npt)
        circle <- rbind(cos(angle), sin(angle))
        coords <- t(eig$vectors %*% (radii * circle))
        center <- c(mean(x), mean(y))
        graphics::lines(coords[, 1] + center[1], coords[, 2] + center[2], col = col, lwd = 2)
      }
      for (j in seq_along(lvl)) {
        idx <- which(groups == lvl[j])
        draw_ellipse(xi[idx], yi[idx], ellipse_level, ellipse_n, group_col[j])
      }
      graphics::legend("topright", legend = lvl, col = group_col, pch = 19, bty = "n")
    }
  } else {
    graphics::plot(xi, yi, xlab = paste0("Comp ", comps[1]),
                   ylab = paste0("Comp ", comps[2]), ...)
    if (!is.null(labels)) {
      graphics::text(xi, yi, labels = labels, pos = 3)
    }
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
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' plot_pls_variables(fit)
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
#' @param groups Optional factor or character vector defining groups for
#'   individuals. When supplied, group-specific colours are used and, if
#'   `ellipse = TRUE`, confidence ellipses are drawn for each group.
#' @param ellipse Logical; draw group confidence ellipses when `groups` are
#'   provided.
#' @param ellipse_level Confidence level for group ellipses (between 0 and 1).
#' @param ellipse_n Number of points used to draw each ellipse.
#' @param group_col Optional vector of colours for the groups. Recycled as
#'   needed.
#' @param ... Additional arguments passed to [graphics::plot()].
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' plot_pls_biplot(fit)
plot_pls_biplot <- function(object, comps = c(1L, 2L), scale_variables = 1,
                            circle = TRUE, circle_col = "grey85",
                            arrow_col = "firebrick", groups = NULL,
                            ellipse = TRUE, ellipse_level = 0.95,
                            ellipse_n = 200L, group_col = NULL, ...) {
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
  xi <- scores[, comps[1]]
  yi <- scores[, comps[2]]
  dots <- list(...)
  base_args <- list(
    x = xi,
    y = yi,
    xlab = paste0("Comp ", comps[1]),
    ylab = paste0("Comp ", comps[2]),
    xlim = c(-lim, lim),
    ylim = c(-lim, lim)
  )
  lvl <- NULL
  if (!is.null(groups)) {
    groups <- as.factor(groups)
    if (length(groups) != length(xi)) {
      stop("Length of `groups` must match the number of observations", call. = FALSE)
    }
    lvl <- levels(groups)
    if (is.null(group_col)) {
      group_col <- grDevices::hcl.colors(length(lvl), palette = "Dark2")
    }
    if (length(group_col) < length(lvl)) {
      group_col <- rep_len(group_col, length(lvl))
    }
    if (!"col" %in% names(dots)) {
      base_args$col <- group_col[groups]
    }
  }
  if (!"pch" %in% names(dots)) {
    base_args$pch <- 19
  }
  if (length(dots)) {
    base_args <- utils::modifyList(base_args, dots)
  }
  base_args$col <- base_args$col %||% graphics::par("col")
  base_args$type <- base_args$type %||% "p"
  do.call(graphics::plot.default, base_args)
  # If custom dots included axis/labels etc. they are already handled via ... above.
  # The call to plot.default ensures compatibility with named arguments.
  if (!is.null(groups) && isTRUE(ellipse) && ellipse_level > 0 && ellipse_level < 1) {
    draw_ellipse <- function(x, y, level, npt, col) {
      if (length(x) < 3L) return()
      Sigma <- stats::cov(cbind(x, y))
      if (any(!is.finite(Sigma))) return()
      eig <- tryCatch(base::eigen(Sigma, symmetric = TRUE), error = function(e) NULL)
      if (is.null(eig)) return()
      vals <- pmax(eig$values, 0)
      if (all(vals == 0)) return()
      radii <- sqrt(stats::qchisq(level, df = 2) * vals)
      angle <- seq(0, 2 * pi, length.out = npt)
      circle <- rbind(cos(angle), sin(angle))
      coords <- t(eig$vectors %*% (radii * circle))
      center <- c(mean(x), mean(y))
      graphics::lines(coords[, 1] + center[1], coords[, 2] + center[2], col = col, lwd = 2)
    }
    for (j in seq_along(lvl)) {
      idx <- which(groups == lvl[j])
      draw_ellipse(xi[idx], yi[idx], ellipse_level, ellipse_n, group_col[j])
    }
    graphics::legend("topright", legend = lvl, col = group_col, pch = 19, bty = "n")
  }
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
#' @param max_comp Maximum number of components to consider.
#'
#' @return A data frame with RSS, RMSE, AIC and BIC per component.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' pls_information_criteria(fit, X, y)
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
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r")
#' pls_select_components(fit, X, y)
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

.parallel_map <- function(items, FUN, parallel = c("none", "future"), future_seed = TRUE) {
  parallel <- match.arg(parallel)
  if (identical(parallel, "future")) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("parallel='future' requires the 'future.apply' package", call. = FALSE)
    }
    future.apply::future_lapply(items, FUN, future.seed = future_seed)
  } else {
    lapply(items, FUN)
  }
}

.pseudoinverse <- function(M, tol = sqrt(.Machine$double.eps)) {
  sv <- svd(M)
  if (length(sv$d) == 0L) {
    return(matrix(0, ncol(M), nrow(M)))
  }
  positive <- sv$d > tol * max(sv$d)
  if (!any(positive)) {
    return(matrix(0, ncol(M), nrow(M)))
  }
  sv$v[, positive, drop = FALSE] %*%
    (t(sv$u[, positive, drop = FALSE]) / sv$d[positive])
}
 

#' Cross-validate PLS models
#'
#' @param X Predictor matrix as accepted by [pls_fit()]
#' @param Y Response matrix or vector as accepted by [pls_fit()]
#' @param ncomp Integer; components grid to evaluate.
#' @param folds Number of folds (ignored when `type = "loo"`).
#' @param type Either "kfold" (default) or "loo".
#' @param algorithm Backend algorithm: "simpls", "nipals", "kernelpls" or
#'   "widekernelpls".
#' @param backend Backend passed to [pls_fit()].
#' @param metrics Metrics to compute (subset of "rmse", "mae", "r2").
#' @param seed Optional seed for reproducibility.
#' @param parallel Logical or character; same semantics as in [pls_bootstrap()].
#' @param future_seed Logical or integer; reproducible seeds for parallel evaluation.
#' @param ... Passed to [pls_fit()].
#' @return A list containing per-fold metrics and their summary across folds.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' pls_cross_validate(X, y, ncomp = 2, folds = 3)
pls_cross_validate <- function(X, Y, ncomp, folds = 5L, type = c("kfold", "loo"),
                               algorithm = c("simpls", "nipals", "kernelpls", 
                                             "widekernelpls"), backend = "arma",
                               metrics = c("rmse", "mae", "r2"), seed = NULL,
                               parallel = c("none", "future"), future_seed = TRUE,
                               ...) {
  type <- match.arg(type)
  algorithm <- match.arg(algorithm)
  metrics <- unique(match.arg(metrics, choices = names(.metric_functions), several.ok = TRUE))
  parallel <- match.arg(parallel)
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
  folds_unique <- seq_len(max(fold_ids))
  tasks <- lapply(folds_unique, function(k) {
    list(
      fold = k,
      train_idx = setdiff(seq_len(n), which(fold_ids == k)),
      test_idx  = which(fold_ids == k)
    )
  })
  fold_chunks <- .parallel_map(tasks, function(task) {
    fit <- pls_fit(Xmat[task$train_idx, , drop = FALSE],
                   Ymat[task$train_idx, , drop = FALSE],
                   ncomp = ncomp, algorithm = algorithm, backend = backend,
                   scores = "none", mode = mode)
    res <- lapply(seq_len(ncomp), function(comp) {
      preds <- predict(fit, Xmat[task$test_idx, , drop = FALSE], ncomp = comp)
      preds <- if (is.vector(preds)) matrix(preds, ncol = 1L) else as.matrix(preds)
      truth <- Ymat[task$test_idx, , drop = FALSE]
      metric_vals <- .compute_metrics(preds, truth, metrics)
      data.frame(
        fold = task$fold,
        ncomp = comp,
        metric = metrics,
        value = unlist(metric_vals),
        row.names = NULL
      )
    })
    do.call(rbind, res)
  }, parallel = parallel, future_seed = future_seed)
  fold_df <- do.call(rbind, fold_chunks)
  summary_df <- stats::aggregate(value ~ ncomp + metric, data = fold_df, FUN = mean)
  list(details = fold_df, summary = summary_df)
}

#' Bootstrap a PLS model
#'
#' Draw bootstrap replicates of a fitted PLS model, refitting on each resample.
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
#' @param type Character; bootstrap scheme, e.g. `"pairs"`, `"residual"`, or `"parametric"`.
#' @param parallel Logical or character; if `TRUE` or one of
#'   `c("sequential", "multisession", "multicore")`, uses the future framework.
#' @param future_seed Logical or integer; forwarded to `future.seed` for
#'   reproducible parallel streams.
#' @param return_scores Logical; if `TRUE`, return component scores for each replicate
#'   (may be large).
#' @param ... Additional arguments forwarded to [pls_fit()].
#' @return A list with bootstrap estimates and summaries.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' pls_bootstrap(X, y, ncomp = 2, R = 20)
pls_bootstrap <- function(X, Y, ncomp, R = 100L, algorithm = c("simpls", "nipals", "kernelpls", "widekernelpls"),
                          backend = "arma", conf = 0.95, seed = NULL,
                          type = c("xy", "xt"), parallel = c("none", "future"),
                          future_seed = TRUE,
                          return_scores = FALSE,
                          ...) {
  algorithm <- match.arg(algorithm)
  type <- match.arg(type)
  parallel <- match.arg(parallel)
  Xmat <- as.matrix(X)
  Ymat <- if (is.vector(Y)) matrix(Y, ncol = 1L) else as.matrix(Y)
  n <- nrow(Xmat)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  mode <- if (ncol(Ymat) == 1L) "pls1" else "pls2"
  base_fit <- pls_fit(Xmat, Ymat, ncomp = ncomp, algorithm = algorithm,
                      backend = backend, mode = mode,
                      scores = if (return_scores || identical(type, "xt")) "r" else "none")
  coef_samples <- vector("list", R)
  score_samples <- if (isTRUE(return_scores)) vector("list", R) else NULL
  resample_indices <- replicate(R, sample.int(n, n, replace = TRUE), simplify = FALSE)
  if (identical(type, "xy")) {
    boot_chunks <- .parallel_map(resample_indices, function(idx) {
      fit <- pls_fit(Xmat[idx, , drop = FALSE], Ymat[idx, , drop = FALSE],
                     ncomp = ncomp, algorithm = algorithm, backend = backend,
                     scores = if (isTRUE(return_scores)) "r" else "none", mode = mode, ...)
      coef_mat <- if (inherits(fit$coefficients, "big.matrix")) fit$coefficients[,] else as.matrix(fit$coefficients)
      scores_mat <- if (isTRUE(return_scores) && !is.null(fit$scores)) as.matrix(fit$scores) else NULL
      list(coef = coef_mat, scores = scores_mat)
    }, parallel = parallel, future_seed = future_seed)
    for (b in seq_along(boot_chunks)) {
      coef_samples[[b]] <- boot_chunks[[b]]$coef
      if (isTRUE(return_scores)) score_samples[[b]] <- boot_chunks[[b]]$scores
    }
  } else {
    # Conditional bootstrap on latent components (X, T)
    scores_base <- as.matrix(base_fit$scores)
    if (is.null(scores_base)) {
      stop("Base fit must include scores for type = 'xt'. Refit with scores = 'r'.", call. = FALSE)
    }
    W <- as.matrix(base_fit$x_weights)
    P <- as.matrix(base_fit$x_loadings)
    if (is.null(W) || is.null(P)) {
      stop("Base fit is missing loadings/weights required for XT bootstrap", call. = FALSE)
    }
    W_eff <- {
      Rmat <- crossprod(P, W)
      Rinv <- tryCatch(solve(Rmat), error = function(e) .pseudoinverse(Rmat))
      W %*% Rinv
    }
    run_boot_xt <- function(idx) {
      Tb <- scores_base[idx, , drop = FALSE]
      Xb <- Xmat[idx, , drop = FALSE]
      Yb <- Ymat[idx, , drop = FALSE]
      XtX <- crossprod(Tb)
      XtX_inv <- tryCatch(solve(XtX), error = function(e) .pseudoinverse(XtX))
      Qt <- XtX_inv %*% crossprod(Tb, Yb)
      coef_mat <- W_eff %*% Qt
      list(coef = coef_mat,
           scores = if (isTRUE(return_scores)) Tb else NULL,
           loadings = t(Qt))
    }
    boot_chunks <- .parallel_map(resample_indices, run_boot_xt,
                                 parallel = parallel, future_seed = future_seed)
    for (b in seq_along(boot_chunks)) {
      coef_samples[[b]] <- boot_chunks[[b]]$coef
      if (isTRUE(return_scores)) score_samples[[b]] <- boot_chunks[[b]]$scores
    }
  }
  coef_arr <- simplify2array(coef_samples)
  alpha <- (1 - conf) / 2
  lower <- apply(coef_arr, 1:2, stats::quantile, probs = alpha)
  upper <- apply(coef_arr, 1:2, stats::quantile, probs = 1 - alpha)
  mean_coef <- apply(coef_arr, 1:2, mean)
  result <- list(mean = mean_coef, lower = lower, upper = upper,
                 samples = coef_samples, type = type,
                 base_fit = base_fit)
  base_coef <- if (inherits(base_fit$coefficients, "big.matrix")) base_fit$coefficients[,] else as.matrix(base_fit$coefficients)
  boot_mat <- sapply(coef_samples, function(mat) as.vector(mat))
  compute_bca <- function(boot_vec, jack_vec, theta_hat) {
    eps <- .Machine$double.eps
    prop_less <- mean(boot_vec < theta_hat)
    prop_less <- min(max(prop_less, eps), 1 - eps)
    z0 <- stats::qnorm(prop_less)
    jack_mean <- mean(jack_vec)
    num <- sum((jack_mean - jack_vec)^3)
    den <- 6 * (sum((jack_mean - jack_vec)^2)^(3/2))
    a <- if (den == 0 || !is.finite(den)) 0 else num / den
    zalpha <- stats::qnorm(c(alpha, 1 - alpha))
    adj <- z0 + (z0 + zalpha) / pmax(1 - a * (z0 + zalpha), eps)
    adj <- stats::pnorm(adj)
    adj <- pmin(pmax(adj, 0), 1)
    c(stats::quantile(boot_vec, probs = adj[1], type = 6, names = FALSE),
      stats::quantile(boot_vec, probs = adj[2], type = 6, names = FALSE))
  }
  jack_sets <- lapply(seq_len(n), function(i) setdiff(seq_len(n), i))
  jack_chunks <- .parallel_map(jack_sets, function(idx) {
    if (identical(type, "xy")) {
      fit <- pls_fit(Xmat[idx, , drop = FALSE], Ymat[idx, , drop = FALSE],
                     ncomp = ncomp, algorithm = algorithm, backend = backend,
                     scores = "none", mode = mode)
      coef_mat <- if (inherits(fit$coefficients, "big.matrix")) fit$coefficients[,] else as.matrix(fit$coefficients)
    } else {
      Tb <- scores_base[idx, , drop = FALSE]
      XtX <- crossprod(Tb)
      XtX_inv <- tryCatch(solve(XtX), error = function(e) .pseudoinverse(XtX))
      Qt <- XtX_inv %*% crossprod(Tb, Ymat[idx, , drop = FALSE])
      coef_mat <- W_eff %*% Qt
    }
    as.vector(coef_mat)
  }, parallel = parallel, future_seed = future_seed)
  jack_mat <- do.call(cbind, jack_chunks)
  base_coef_vec <- as.vector(base_coef)
  bca_bounds <- vapply(seq_len(nrow(boot_mat)), function(i) {
    compute_bca(boot_mat[i, ], jack_mat[i, ], base_coef_vec[i])
  }, numeric(2L))
  bca_lower <- matrix(bca_bounds[1, ], nrow = nrow(base_coef))
  bca_upper <- matrix(bca_bounds[2, ], nrow = nrow(base_coef))
  result$bca_lower <- bca_lower
  result$bca_upper <- bca_upper
  result$jackknife <- jack_mat
  if (isTRUE(return_scores)) {
    result$score_samples <- score_samples
  }
  result
}

#' Summarise bootstrap estimates
#'
#' @param boot_result Result returned by [pls_bootstrap()].
#'
#' @return A data frame containing mean, standard deviation, percentile and
#'   BCa confidence intervals for each coefficient.
#' @export
summarise_pls_bootstrap <- function(boot_result) {
  if (is.null(boot_result$mean) || is.null(boot_result$samples)) {
    stop("`boot_result` does not look like an output from pls_bootstrap()", call. = FALSE)
  }
  mean_mat <- as.matrix(boot_result$mean)
  lower <- as.matrix(boot_result$lower)
  upper <- as.matrix(boot_result$upper)
  bca_lower <- as.matrix(boot_result$bca_lower)
  bca_upper <- as.matrix(boot_result$bca_upper)
  sample_mat <- sapply(boot_result$samples, function(mat) as.vector(mat))
  sd_vec <- apply(sample_mat, 1L, stats::sd)
  vars <- rownames(mean_mat) %||% paste0("X", seq_len(nrow(mean_mat)))
  resp <- colnames(mean_mat) %||% paste0("Y", seq_len(ncol(mean_mat)))
  grid <- expand.grid(variable = vars, response = resp, KEEP.OUT.ATTRS = FALSE)
  data.frame(
    variable = grid$variable,
    response = grid$response,
    mean = as.vector(mean_mat),
    sd = sd_vec,
    percentile_lower = as.vector(lower),
    percentile_upper = as.vector(upper),
    bca_lower = as.vector(bca_lower),
    bca_upper = as.vector(bca_upper),
    row.names = NULL
  )
}

#' Boxplots of bootstrap coefficient distributions
#'
#' @param boot_result Result returned by [pls_bootstrap()].
#' @param responses Optional character vector selecting response columns.
#' @param variables Optional character vector selecting predictor variables.
#' @param ... Additional arguments passed to [graphics::boxplot()].
#' @importFrom graphics boxplot abline
#' @export
plot_pls_bootstrap_coefficients <- function(boot_result, responses = NULL,
                                            variables = NULL, ...) {
  if (is.null(boot_result$samples)) {
    stop("`boot_result` lacks bootstrap samples; call pls_bootstrap() first", call. = FALSE)
  }
  mean_mat <- as.matrix(boot_result$mean)
  vars <- rownames(mean_mat) %||% paste0("X", seq_len(nrow(mean_mat)))
  resp <- colnames(mean_mat) %||% paste0("Y", seq_len(ncol(mean_mat)))
  sample_mat <- sapply(boot_result$samples, function(mat) as.vector(mat))
  grid <- expand.grid(variable = vars, response = resp, KEEP.OUT.ATTRS = FALSE)
  labels <- paste(grid$response, grid$variable, sep = " :: ")
  keep <- rep(TRUE, length(labels))
  if (!is.null(responses)) keep <- keep & grid$response %in% responses
  if (!is.null(variables)) keep <- keep & grid$variable %in% variables
  if (!any(keep)) {
    stop("No coefficients match the requested filters", call. = FALSE)
  }
  splits <- lapply(which(keep), function(i) sample_mat[i, ])
  names(splits) <- labels[keep]
  graphics::boxplot(splits, las = 2, ...)
  graphics::abline(h = 0, col = "grey70", lty = 3)
  invisible(NULL)
}

#' Boxplots of bootstrap score distributions
#'
#' Visualise the variability of latent scores obtained through
#' [pls_bootstrap()] when `return_scores = TRUE`.
#'
#' @param boot_result Result returned by [pls_bootstrap()].
#' @param components Optional vector of component indices or names to include.
#' @param observations Optional vector of observation indices or names to include.
#' @param ... Additional arguments passed to [graphics::boxplot()].
#' @importFrom graphics boxplot
#' @export
plot_pls_bootstrap_scores <- function(boot_result, components = NULL,
                                      observations = NULL, ...) {
  samples <- boot_result$score_samples
  if (is.null(samples)) {
    stop("`boot_result` does not contain bootstrap scores; refit with return_scores = TRUE", call. = FALSE)
  }
  keep <- !vapply(samples, is.null, logical(1L))
  samples <- samples[keep]
  if (!length(samples)) {
    stop("No bootstrap score samples available", call. = FALSE)
  }
  mats <- lapply(samples, function(mat) as.matrix(mat))
  nobs <- nrow(mats[[1]])
  ncomp <- ncol(mats[[1]])
  if (any(vapply(mats, nrow, integer(1L)) != nobs) ||
      any(vapply(mats, ncol, integer(1L)) != ncomp)) {
    stop("Inconsistent score dimensions across bootstrap samples", call. = FALSE)
  }
  comp_names <- colnames(mats[[1]]) %||% paste0("Comp", seq_len(ncomp))
  obs_names <- rownames(mats[[1]]) %||%
    rownames(boot_result$base_fit$scores) %||%
    paste0("Obs", seq_len(nobs))
  
  comp_idx <- if (is.null(components)) {
    seq_len(ncomp)
  } else if (is.numeric(components)) {
    components
  } else {
    match(components, comp_names)
  }
  if (any(!is.finite(comp_idx)) || any(comp_idx < 1L) || any(comp_idx > ncomp)) {
    stop("`components` must reference valid component indices or names", call. = FALSE)
  }
  obs_idx <- if (is.null(observations)) {
    seq_len(nobs)
  } else if (is.numeric(observations)) {
    observations
  } else {
    match(observations, obs_names)
  }
  if (any(!is.finite(obs_idx)) || any(obs_idx < 1L) || any(obs_idx > nobs)) {
    stop("`observations` must reference valid observation indices or names", call. = FALSE)
  }
  
  combos <- expand.grid(obs = obs_idx, comp = comp_idx, KEEP.OUT.ATTRS = FALSE)
  labels <- paste(obs_names[combos$obs], comp_names[combos$comp], sep = " :: ")
  splits <- lapply(seq_len(nrow(combos)), function(i) {
    obs_i <- combos$obs[i]
    comp_i <- combos$comp[i]
    vapply(mats, function(mat) mat[obs_i, comp_i], numeric(1L))
  })
  names(splits) <- labels
  graphics::boxplot(splits, las = 2, ...)
  invisible(NULL)
}

#' Naive sparsity control by coefficient thresholding
#'
#' @param object A fitted PLS model.
#' @param threshold Values below this absolute magnitude are set to zero.
#' @return A modified copy of `object` with thresholded coefficients.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(40), nrow = 10)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2)
#' pls_threshold(fit, threshold = 0.05)
pls_threshold <- function(object, threshold) {
  if (is.null(object$coefficients)) {
    stop("Object does not contain coefficient estimates", call. = FALSE)
  }
  if (!is.numeric(threshold) || length(threshold) != 1L || !is.finite(threshold) || threshold < 0) {
    stop("`threshold` must be a single non-negative numeric value", call. = FALSE)
  }
  if (inherits(object$coefficients, "big.matrix")) {
    coef_mat <- object$coefficients[,]
    coef_mat[abs(coef_mat) < threshold] <- 0
    object$coefficients[,] <- coef_mat
  } else {
    coef_mat <- as.matrix(object$coefficients)
    coef_mat[abs(coef_mat) < threshold] <- 0
    object$coefficients <- coef_mat
  }
  object$coef_threshold <- threshold
  object
}

#' Select components from cross-validation results
#'
#' @param cv_result Result returned by [pls_cross_validate()].
#' @param metric Metric to optimise.
#' @param minimise Logical; whether the metric should be minimised.
#' @return Selected number of components.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' cv <- pls_cross_validate(X, y, ncomp = 2, folds = 3)
#' pls_cv_select(cv, metric = "rmse")
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
