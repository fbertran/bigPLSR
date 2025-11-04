test_that("PLS solvers agree on synthetic data", {
  skip_if_not_installed("bigmemory")
  set.seed(42)
  n <- 120
  p <- 6
  X <- matrix(rnorm(n * p), n, p)
  beta <- seq_len(p) / p
  y <- drop(X %*% beta + rnorm(n, sd = 0.1))

  X_bm <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
  y_bm <- bigmemory::big.matrix(nrow = n, ncol = 1, type = "double")
  for (j in seq_len(p)) {
    X_bm[, j] <- X[, j]
  }
  y_bm[, 1] <- y

  fit_direct_nipals <- pls1_dense_a(X_bm, y_bm, ncomp = 3, center = TRUE,
                                         scale = TRUE, algorithm = "nipals")
  fit_stream_nipals <- pls1_stream_a(X_bm, y_bm, ncomp = 3, chunk_size = 25,
                                         center = TRUE, scale = TRUE,
                                         algorithm = "nipals")
  
  expect_equal(fit_direct_nipals$ncomp, fit_stream_nipals$ncomp)
  expect_equal(fit_direct_nipals$coefficients, fit_stream_nipals$coefficients,
               tolerance = 1e-6)
  expect_equal(fit_direct_nipals$intercept, fit_stream_nipals$intercept,
               tolerance = 1e-6)
  
  fit_direct_simpls <- pls1_dense_a(X_bm, y_bm, ncomp = 3, center = TRUE,
                                         scale = FALSE, algorithm = "simpls")
  fit_stream_simpls <- pls1_stream_a(X_bm, y_bm, ncomp = 3, chunk_size = 25,
                                         center = TRUE, scale = FALSE,
                                         algorithm = "simpls")
  
  expect_equal(fit_direct_simpls$ncomp, fit_stream_simpls$ncomp)
  expect_equal(fit_direct_simpls$coefficients, fit_stream_simpls$coefficients,
               tolerance = 1e-6)
  expect_equal(fit_direct_simpls$intercept, fit_stream_simpls$intercept,
               tolerance = 1e-6)
  
  expect_equal(fit_direct_simpls$coefficients[,], fit_direct_nipals$coefficients,
               tolerance = 1e-6)
  expect_equal(fit_direct_simpls$intercept, fit_direct_nipals$intercept,
               tolerance = 1e-6)
})

test_that("numeric responses are accepted", {
  skip_if_not_installed("bigmemory")
  set.seed(1)
  X <- matrix(rnorm(40), 20, 2)
  y <- rnorm(20)
  X_bm <- bigmemory::big.matrix(nrow = 20, ncol = 2, type = "double")
  for (j in 1:2) {
    X_bm[, j] <- X[, j]
  }
  expect_no_error({
    fit <- pls1_dense(X_bm, y, ncomp = 2, algorithm = "simpls")
    expect_true(is.list(fit))
    expect_length(fit$coefficients, 2)
    fit2 <- pls1_dense_a(X_bm, y, ncomp = 2, algorithm = "simpls")
    expect_true(is.list(fit2))
    expect_length(fit2$coefficients, 2)
    fit3 <- pls1_dense_ya(X_bm, y, ncomp = 2, algorithm = "simpls")
    expect_true(is.list(fit3))
    expect_length(fit3$coefficients, 2)
  })
})



test_that("SIMPLS and NIPALS agree for multivariate responses", {
  skip_if_not_installed("bigmemory")
  set.seed(2024)
  n <- 80
  p <- 6
  q <- 3
  X <- matrix(rnorm(n * p), n, p)
  true_coef <- matrix(runif(p * q, -1, 1), p, q)
  Y <- X %*% true_coef + matrix(rnorm(n * q, sd = 0.05), n, q)
  
  X_bm <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
  Y_bm <- bigmemory::big.matrix(nrow = n, ncol = q, type = "double")
  for (j in seq_len(p)) {
    X_bm[, j] <- X[, j]
  }
  for (k in seq_len(q)) {
    Y_bm[, k] <- Y[, k]
  }
  
  fit_simpls <- pls2_dense(X_bm, Y_bm, ncomp = 3, center = TRUE, scale = TRUE,
                         algorithm = "simpls")
  fit_nipals <- pls2_dense(X_bm, Y_bm, ncomp = 3, center = TRUE, scale = TRUE,
                         algorithm = "nipals")
  
  expect_equal(fit_simpls$coefficients, fit_nipals$coefficients, tolerance = 1e-6)
  expect_equal(fit_simpls$intercept, fit_nipals$intercept, tolerance = 1e-6)
  
  fit_stream_simpls <- pls2_stream(X_bm, Y_bm, ncomp = 3, center = TRUE,
                                       scale = TRUE, block_size = 16,
                                       algorithm = "simpls")
  fit_stream_nipals <- pls2_stream(X_bm, Y_bm, ncomp = 3, center = TRUE,
                                       scale = TRUE, block_size = 16,
                                       algorithm = "nipals")
  
  expect_equal(fit_stream_simpls$coefficients, fit_simpls$coefficients, tolerance = 1e-6)
  expect_equal(fit_stream_simpls$intercept, fit_simpls$intercept, tolerance = 1e-6)
  expect_equal(fit_stream_nipals$coefficients, fit_nipals$coefficients, tolerance = 1e-6)
  expect_equal(fit_stream_nipals$intercept, fit_nipals$intercept, tolerance = 1e-6)
})
