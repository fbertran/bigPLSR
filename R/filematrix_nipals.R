# Internal R-level NIPALS backend for filematrix inputs.
#
# This intentionally reads row blocks through the filematrix provider instead
# of using bigmemory mmap access. It mirrors the existing streamed NIPALS math
# and avoids forming X'X; each pass keeps only factor matrices and row chunks.
run_filematrix_nipals <- function(X, y, ncomp, mode, chunk_size, scores,
                                  tol = 1e-8,
                                  max_iter = getOption("bigPLSR.filematrix.nipals.max_iter", 500L),
                                  scores_name = "scores",
                                  scores_target = c("auto", "new", "existing"),
                                  scores_bm = NULL,
                                  scores_backingfile = NULL,
                                  scores_backingpath = NULL,
                                  scores_descriptorfile = NULL,
                                  ...) {
  if (!requireNamespace("filematrix", quietly = TRUE)) {
    stop("backend='filematrix' requires package 'filematrix'. Install it first.", call. = FALSE)
  }

  mode <- match.arg(mode, c("pls1", "pls2"))
  scores <- match.arg(scores, c("none", "r", "big"))
  scores_target <- match.arg(scores_target)

  if (!is_filematrix_object(X)) {
    stop("For backend='filematrix', X must be a filematrix object.", call. = FALSE)
  }

  chunk_size <- .bigPLSR_filematrix_positive_int(chunk_size, "chunk_size")
  ncomp <- .bigPLSR_filematrix_positive_int(ncomp, "ncomp")
  tol <- as.numeric(tol)[1L]
  if (!is.finite(tol) || tol <= 0) {
    stop("tol must be a positive finite numeric value.", call. = FALSE)
  }
  max_iter <- .bigPLSR_filematrix_positive_int(max_iter, "max_iter")

  x_provider <- make_filematrix_row_provider(X, chunk_size = chunk_size)
  y_provider <- .bigPLSR_make_filematrix_response_provider(y, chunk_size = chunk_size)

  n <- x_provider$nrow()
  p <- x_provider$ncol()
  q <- y_provider$ncol()

  if (y_provider$nrow() != n) {
    stop("X and y must have the same number of rows.", call. = FALSE)
  }
  if (mode == "pls1" && q != 1L) {
    stop("mode='pls1' requires y to have one column", call. = FALSE)
  }

  ncomp <- min(ncomp, p)
  starts <- seq.int(1L, n, by = chunk_size)

  x_means <- numeric(p)
  y_means <- numeric(q)
  for (i0 in starts) {
    i1 <- min(n, i0 + chunk_size - 1L)
    x_block <- x_provider$get_rows(i0, i1)
    y_block <- y_provider$get_rows(i0, i1)
    x_means <- x_means + colSums(x_block)
    y_means <- y_means + colSums(y_block)
  }
  x_means <- x_means / n
  y_means <- y_means / n

  W <- matrix(0, nrow = p, ncol = ncomp)
  P <- matrix(0, nrow = p, ncol = ncomp)
  Q <- matrix(0, nrow = q, ncol = ncomp)
  B <- numeric(ncomp)
  scores_mat <- if (identical(scores, "none")) NULL else matrix(0, nrow = n, ncol = ncomp)

  eps <- 1e-12
  eps_sq <- 1e-20
  actual_comp <- 0L

  deflated_block <- function(i0, i1, used_components) {
    x_block <- x_provider$get_rows(i0, i1)
    y_block <- y_provider$get_rows(i0, i1)
    x_block <- sweep(x_block, 2L, x_means, FUN = "-")
    y_block <- sweep(y_block, 2L, y_means, FUN = "-")

    if (used_components > 0L) {
      for (h in seq_len(used_components)) {
        t_h <- drop(x_block %*% W[, h, drop = FALSE])
        x_block <- x_block - tcrossprod(t_h, P[, h])
        y_block <- y_block - tcrossprod(B[h] * t_h, Q[, h])
      }
    }

    list(x = x_block, y = y_block)
  }

  for (a in seq_len(ncomp)) {
    sumsq_y <- numeric(q)
    for (i0 in starts) {
      i1 <- min(n, i0 + chunk_size - 1L)
      block <- deflated_block(i0, i1, actual_comp)
      sumsq_y <- sumsq_y + colSums(block$y * block$y)
    }

    chosen_col <- which.max(sumsq_y)
    if (!length(chosen_col) || sumsq_y[[chosen_col]] <= eps_sq) {
      break
    }

    u <- numeric(n)
    for (i0 in starts) {
      i1 <- min(n, i0 + chunk_size - 1L)
      block <- deflated_block(i0, i1, actual_comp)
      u[i0:i1] <- block$y[, chosen_col]
    }

    w <- numeric(p)
    t_scores <- numeric(n)
    c_vec <- numeric(q)
    converged <- FALSE

    for (iter in seq_len(max_iter)) {
      u_prev <- u
      u_norm_sq <- sum(u * u)
      if (u_norm_sq <= eps_sq) {
        break
      }

      w <- numeric(p)
      for (i0 in starts) {
        i1 <- min(n, i0 + chunk_size - 1L)
        block <- deflated_block(i0, i1, actual_comp)
        w <- w + drop(crossprod(block$x, u[i0:i1]))
      }

      w <- w / u_norm_sq
      w_norm <- sqrt(sum(w * w))
      if (w_norm <= eps) {
        break
      }
      w <- w / w_norm

      t_scores <- numeric(n)
      q_acc <- numeric(q)
      t_norm_sq <- 0
      for (i0 in starts) {
        i1 <- min(n, i0 + chunk_size - 1L)
        block <- deflated_block(i0, i1, actual_comp)
        t_block <- drop(block$x %*% w)
        t_scores[i0:i1] <- t_block
        t_norm_sq <- t_norm_sq + sum(t_block * t_block)
        q_acc <- q_acc + drop(crossprod(block$y, t_block))
      }

      if (t_norm_sq <= eps_sq) {
        break
      }

      c_vec <- q_acc / t_norm_sq
      c_norm_sq <- sum(c_vec * c_vec)
      if (c_norm_sq <= eps_sq) {
        break
      }

      u_new <- numeric(n)
      u_new_norm_sq <- 0
      for (i0 in starts) {
        i1 <- min(n, i0 + chunk_size - 1L)
        block <- deflated_block(i0, i1, actual_comp)
        u_block <- drop(block$y %*% c_vec) / c_norm_sq
        u_new[i0:i1] <- u_block
        u_new_norm_sq <- u_new_norm_sq + sum(u_block * u_block)
      }

      diff_norm <- sqrt(sum((u_new - u_prev) * (u_new - u_prev)))
      denom <- max(1, sqrt(u_new_norm_sq), sqrt(sum(u_prev * u_prev)))
      u <- u_new

      if (diff_norm / denom < tol) {
        converged <- TRUE
        break
      }
    }

    if (!converged && sqrt(sum(u * u)) <= eps) {
      break
    }

    t_norm_sq <- sum(t_scores * t_scores)
    if (t_norm_sq <= eps_sq) {
      break
    }

    p_vec <- numeric(p)
    for (i0 in starts) {
      i1 <- min(n, i0 + chunk_size - 1L)
      block <- deflated_block(i0, i1, actual_comp)
      p_vec <- p_vec + drop(crossprod(block$x, t_scores[i0:i1]))
    }
    p_vec <- p_vec / t_norm_sq

    b_scalar <- sum(t_scores * u) / t_norm_sq

    W[, a] <- w
    P[, a] <- p_vec
    Q[, a] <- c_vec
    B[[a]] <- b_scalar
    if (!is.null(scores_mat)) {
      scores_mat[, a] <- t_scores
    }
    actual_comp <- actual_comp + 1L
  }

  if (actual_comp == 0L) {
    return(list(
      coefficients = NULL,
      intercept = NULL,
      x_weights = NULL,
      weights = NULL,
      x_loadings = NULL,
      loadings = NULL,
      y_loadings = NULL,
      scores = NULL,
      x_means = x_means,
      y_means = y_means,
      x_scales = rep(1, p),
      y_scales = rep(1, q),
      B = numeric(),
      ncomp = 0L,
      mode = mode,
      backend = "filematrix"
    ))
  }

  W_used <- W[, seq_len(actual_comp), drop = FALSE]
  P_used <- P[, seq_len(actual_comp), drop = FALSE]
  Q_used <- Q[, seq_len(actual_comp), drop = FALSE]
  B_used <- B[seq_len(actual_comp)]

  ptw <- crossprod(P_used, W_used)
  ptw_inv <- tryCatch(
    solve(ptw),
    error = function(e) {
      ridge <- getOption("bigPLSR.filematrix.proj.ridge", 1e-10)
      solve(ptw + diag(ridge, nrow(ptw), ncol(ptw)))
    }
  )
  coef_internal <- W_used %*% ptw_inv %*% diag(B_used, nrow = actual_comp, ncol = actual_comp) %*% t(Q_used)
  intercept <- as.numeric(y_means - drop(x_means %*% coef_internal))

  scores_out <- NULL
  if (!is.null(scores_mat)) {
    scores_out <- scores_mat[, seq_len(actual_comp), drop = FALSE]
    if (identical(scores, "big")) {
      scores_out <- .bigPLSR_filematrix_scores_to_big(
        scores_out,
        scores_target = scores_target,
        scores_bm = scores_bm,
        scores_backingfile = scores_backingfile,
        scores_backingpath = scores_backingpath,
        scores_descriptorfile = scores_descriptorfile
      )
    }
  }

  list(
    coefficients = coef_internal,
    intercept = intercept,
    x_weights = W_used,
    weights = W_used,
    x_loadings = P_used,
    loadings = P_used,
    y_loadings = Q_used,
    scores = scores_out,
    x_means = x_means,
    y_means = y_means,
    x_scales = rep(1, p),
    y_scales = rep(1, q),
    B = B_used,
    ncomp = actual_comp,
    mode = mode,
    backend = "filematrix",
    chunk_size = chunk_size
  )
}

.bigPLSR_make_filematrix_response_provider <- function(y, chunk_size) {
  if (is_filematrix_object(y)) {
    return(make_filematrix_row_provider(y, chunk_size = chunk_size))
  }

  if (inherits(y, "sparseMatrix")) {
    stop("For backend='filematrix', sparse response matrices are not supported.", call. = FALSE)
  }

  y_mat <- if (is.null(dim(y))) {
    matrix(y, ncol = 1L)
  } else {
    as.matrix(y)
  }

  if (!is.numeric(y_mat)) {
    stop("For backend='filematrix', y must be numeric, filematrix-backed, or coercible to a numeric matrix.", call. = FALSE)
  }
  storage.mode(y_mat) <- "double"

  nr <- nrow(y_mat)
  nc <- ncol(y_mat)
  .validate_provider_dim(nr, "nrow(y)")
  .validate_provider_dim(nc, "ncol(y)")

  provider <- list(
    nrow = function() nr,
    ncol = function() nc,
    get_rows = function(i0, i1) {
      i0 <- .bigPLSR_filematrix_positive_int(i0, "i0")
      i1 <- .bigPLSR_filematrix_positive_int(i1, "i1")
      .validate_closed_interval(i0, i1, nr, "row")
      y_mat[seq.int(i0, i1), , drop = FALSE]
    },
    storage_type = "in_memory"
  )
  class(provider) <- c("matrix_row_provider", "row_block_provider")
  provider
}

.bigPLSR_filematrix_scores_to_big <- function(scores_mat, scores_target,
                                              scores_bm = NULL,
                                              scores_backingfile = NULL,
                                              scores_backingpath = NULL,
                                              scores_descriptorfile = NULL) {
  if (identical(scores_target, "existing") &&
      is.null(scores_bm) &&
      is.null(scores_backingfile)) {
    stop("scores_target='existing' requires scores_bm or backingfile/path/descriptorfile", call. = FALSE)
  }

  if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix")) {
    scores_bm[,] <- scores_mat
    return(scores_bm)
  }

  if (!is.null(scores_bm) && inherits(scores_bm, "big.matrix.descriptor")) {
    bm <- bigmemory::attach.big.matrix(scores_bm)
    bm[,] <- scores_mat
    return(bm)
  }

  if (!is.null(scores_backingfile)) {
    default_path <- getOption("bigPLSR.backingpath_default", tempdir())
    bm <- bigmemory::filebacked.big.matrix(
      nrow = nrow(scores_mat),
      ncol = ncol(scores_mat),
      type = "double",
      backingfile = scores_backingfile,
      backingpath = scores_backingpath %||% default_path,
      descriptorfile = scores_descriptorfile %||% "scores.desc"
    )
    bm[,] <- scores_mat
    return(bm)
  }

  scores_mat
}

.bigPLSR_filematrix_positive_int <- function(x, label) {
  if (length(x) != 1L || is.na(x) || !is.finite(x) || x < 1L || x != as.integer(x)) {
    stop(label, " must be a positive scalar integer.", call. = FALSE)
  }
  as.integer(x)
}
