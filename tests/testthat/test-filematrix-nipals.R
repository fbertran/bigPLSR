.make_filematrix_test_matrix <- function(x, prefix) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  base <- file.path(
    tempdir(),
    paste(prefix, Sys.getpid(), sample.int(100000000L, 1L), sep = "_")
  )
  fm <- filematrix::fm.create(filenamebase = base, nrow = nrow(x), ncol = ncol(x), type = "double")
  fm[,] <- x
  fm
}

test_that("filematrix NIPALS PLS1 fits a tiny file-backed problem", {
  skip_if_not_installed("filematrix")

  set.seed(201)
  n <- 40
  p <- 12
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(1.4 * X[, 1] - 0.6 * X[, 4] + rnorm(n, sd = 0.1), n, 1)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_pls1_X")
  fmY <- .make_filematrix_test_matrix(Y, "bigplsr_fm_pls1_Y")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)
  on.exit(filematrix::closeAndDeleteFiles(fmY), add = TRUE)

  fit <- pls_fit(
    fmX, fmY,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls1",
    scores = "none",
    chunk_size = 11
  )

  coef <- as.matrix(fit$coefficients)
  expect_equal(fit$backend, "filematrix")
  expect_equal(fit$mode, "pls1")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(dim(coef), c(p, 1L))
  expect_true(all(is.finite(coef)))
  expect_gt(fit$ncomp, 0)
  expect_lte(fit$ncomp, 2)
  expect_null(fit$scores)

  ref <- pls_fit(X, Y, ncomp = 2, backend = "arma", algorithm = "nipals", mode = "pls1", scores = "none")
  expect_gt(cor(drop(pls_predict_response(fit, X)), drop(pls_predict_response(ref, X))), 0.9)
})

test_that("filematrix NIPALS PLS2 fits a tiny multi-response problem", {
  skip_if_not_installed("filematrix")

  set.seed(202)
  n <- 45
  p <- 10
  q <- 3
  X <- matrix(rnorm(n * p), n, p)
  B <- matrix(rnorm(p * q), p, q)
  Y <- X %*% B + matrix(rnorm(n * q, sd = 0.05), n, q)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_pls2_X")
  fmY <- .make_filematrix_test_matrix(Y, "bigplsr_fm_pls2_Y")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)
  on.exit(filematrix::closeAndDeleteFiles(fmY), add = TRUE)

  fit <- pls_fit(
    fmX, fmY,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls2",
    scores = "none",
    chunk_size = 13
  )

  coef <- as.matrix(fit$coefficients)
  expect_equal(fit$backend, "filematrix")
  expect_equal(fit$mode, "pls2")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(dim(coef), c(p, q))
  expect_true(all(is.finite(coef)))
  expect_null(fit$scores)

  ref <- pls_fit(X, Y, ncomp = 2, backend = "arma", algorithm = "nipals", mode = "pls2", scores = "none")
  pred_fit <- pls_predict_response(fit, X)
  pred_ref <- pls_predict_response(ref, X)
  pred_cor <- vapply(seq_len(q), function(j) cor(pred_fit[, j], pred_ref[, j]), numeric(1))
  expect_true(all(pred_cor > 0.85))
})

test_that("filematrix NIPALS preserves explicit PLS2 mode for one-column response", {
  skip_if_not_installed("filematrix")

  set.seed(203)
  n <- 35
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(X[, 2] - 0.3 * X[, 5] + rnorm(n, sd = 0.1), n, 1)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_pls2_one_X")
  fmY <- .make_filematrix_test_matrix(Y, "bigplsr_fm_pls2_one_Y")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)
  on.exit(filematrix::closeAndDeleteFiles(fmY), add = TRUE)

  fit <- pls_fit(
    fmX, fmY,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls2",
    scores = "none",
    chunk_size = 9
  )

  expect_equal(fit$mode, "pls2")
  expect_equal(fit$backend, "filematrix")
  expect_equal(dim(as.matrix(fit$coefficients)), c(p, 1L))
  expect_null(fit$scores)
})

test_that("filematrix NIPALS can return in-memory scores", {
  skip_if_not_installed("filematrix")

  set.seed(204)
  n <- 30
  p <- 9
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(0.8 * X[, 1] + 0.5 * X[, 3] + rnorm(n, sd = 0.15), n, 1)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_scores_X")
  fmY <- .make_filematrix_test_matrix(Y, "bigplsr_fm_scores_Y")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)
  on.exit(filematrix::closeAndDeleteFiles(fmY), add = TRUE)

  fit <- pls_fit(
    fmX, fmY,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls1",
    scores = "r",
    chunk_size = 7
  )

  expect_true(is.matrix(fit$scores))
  expect_equal(nrow(fit$scores), n)
  expect_equal(ncol(fit$scores), fit$ncomp)
  expect_true(all(is.finite(fit$scores)))
})

test_that("filematrix NIPALS accepts an in-memory response", {
  skip_if_not_installed("filematrix")

  set.seed(206)
  n <- 32
  p <- 7
  X <- matrix(rnorm(n * p), n, p)
  y <- 0.9 * X[, 1] - 0.4 * X[, 2] + rnorm(n, sd = 0.1)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_mem_y_X")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)

  fit <- pls_fit(
    fmX, y,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls1",
    scores = "none",
    chunk_size = 8
  )

  expect_equal(fit$backend, "filematrix")
  expect_equal(fit$mode, "pls1")
  expect_equal(dim(as.matrix(fit$coefficients)), c(p, 1L))
  expect_null(fit$scores)
})

test_that("filematrix NIPALS handles a modest wide predictor matrix without X cross-product materialization", {
  skip_if_not_installed("filematrix")

  set.seed(205)
  n <- 25
  p <- 1200
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(0.7 * X[, 1] - 0.4 * X[, 120] + rnorm(n, sd = 0.2), n, 1)

  fmX <- .make_filematrix_test_matrix(X, "bigplsr_fm_wide_X")
  fmY <- .make_filematrix_test_matrix(Y, "bigplsr_fm_wide_Y")
  on.exit(filematrix::closeAndDeleteFiles(fmX), add = TRUE)
  on.exit(filematrix::closeAndDeleteFiles(fmY), add = TRUE)

  fit <- pls_fit(
    fmX, fmY,
    ncomp = 2,
    backend = "filematrix",
    algorithm = "nipals",
    mode = "pls1",
    scores = "none",
    chunk_size = 5
  )

  expect_equal(fit$backend, "filematrix")
  expect_equal(fit$mode, "pls1")
  expect_equal(dim(as.matrix(fit$coefficients)), c(p, 1L))
  expect_true(all(is.finite(fit$coefficients)))
  expect_null(fit$scores)
})
