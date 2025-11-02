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
  
  fit_direct <- pls_bigmemory_fit(X_bm, y_bm, ncomp = 3, center = TRUE, scale = TRUE)
  fit_stream <- pls_streaming_fit(X_bm, y_bm, ncomp = 3, block_size = 25,
                                  center = TRUE, scale = TRUE)
  
  expect_equal(fit_direct$ncomp, fit_stream$ncomp)
  expect_equal(fit_direct$coefficients, fit_stream$coefficients, tolerance = 1e-6)
  expect_equal(fit_direct$intercept, fit_stream$intercept, tolerance = 1e-6)
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
    fit <- pls_bigmemory_fit(X_bm, y, ncomp = 2)
    expect_true(is.list(fit))
    expect_length(fit$coefficients, 2)
  })
})
