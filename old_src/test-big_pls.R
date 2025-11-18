test_that("big_pls_fit approximates OLS when enough components are used", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bigmemory")

  set.seed(123)
  n <- 30
  p <- 5
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- runif(p, -1, 1)
  y <- as.numeric(X %*% beta + rnorm(n, sd = 0.01))

  bm <- bigmemory::as.big.matrix(X)
  fit <- pls1_dense(bm, y, ncomp = p, center = TRUE, scale = FALSE,
                     center_y = TRUE, scale_y = FALSE , algorithm = "simpls")

  expect_equal(length(fit$coefficients), p)
  preds <- drop(cbind(1, X) %*% c(fit$intercept, fit$coefficients))
  expect_lt(max(abs(preds - y)), 0.25)
})

test_that("streaming fit matches direct fit", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bigmemory")

  set.seed(321)
  n <- 40
  p <- 6
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- rnorm(p)
  y <- as.numeric(X %*% beta + rnorm(n, sd = 0.05))

  bm <- bigmemory::as.big.matrix(X)
  direct <- pls1_dense(bm, y, ncomp = 4, algorithm = "simpls")
  streaming <- pls1_stream(bm, y, ncomp = 4, chunk_size = 7, algorithm = "simpls")

  expect_equal(streaming$ncomp, direct$ncomp)
  expect_equal(streaming$intercept, direct$intercept, tolerance = 1e-8)
  expect_equal(streaming$coefficients, direct$coefficients, tolerance = 1e-8)
})


test_that("dense big_pls algorithms agree", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bigmemory")
  
  set.seed(2024)
  n <- 25
  p <- 4
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- seq_len(p) / p
  y <- drop(X %*% beta + rnorm(n, sd = 0.05))
  
  X_bm <- bigmemory::as.big.matrix(X)
  y_bm <- bigmemory::as.big.matrix(matrix(y, ncol = 1))
  
  simpls_fit <- pls1_dense_a(X_bm, y_bm, ncomp = 3, algorithm = "simpls")
  nipals_fit <- pls1_dense_a(X_bm, y_bm, ncomp = 3, algorithm = "nipals")
  expect_equal(nipals_fit$ncomp, simpls_fit$ncomp)
  expect_equal(sort(names(nipals_fit)), sort(names(simpls_fit)))
  expect_equal(nipals_fit$coefficients, simpls_fit$coefficients, tolerance = 1e-6)
  expect_equal(nipals_fit$x_center, simpls_fit$x_center, tolerance = 1e-6)
  expect_equal(nipals_fit$x_scale, simpls_fit$x_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$y_center, simpls_fit$y_center, tolerance = 1e-6)
  expect_equal(nipals_fit$y_scale, simpls_fit$y_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$intercept, simpls_fit$intercept, tolerance = 1e-6)
  
  simpls_stream <- pls1_stream_a(X_bm, y_bm, ncomp = 3, chunk_size = 8,
                                  algorithm = "simpls")
  nipals_stream <- pls1_stream_a(X_bm, y_bm, ncomp = 3, chunk_size = 8,
                                  algorithm = "nipals")
  expect_equal(nipals_stream$ncomp, simpls_stream$ncomp)
  expect_equal(sort(names(nipals_stream)), sort(names(simpls_stream)))
  expect_equal(nipals_stream$coefficients, simpls_stream$coefficients, tolerance = 1e-6)
  expect_equal(nipals_stream$x_center, simpls_stream$x_center, tolerance = 1e-6)
  expect_equal(nipals_stream$x_scale, simpls_stream$x_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$y_center, simpls_stream$y_center, tolerance = 1e-6)
  expect_equal(nipals_stream$y_scale, simpls_stream$y_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$intercept, simpls_stream$intercept, tolerance = 1e-6)
})


test_that("big memory interfaces support both algorithms", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bigmemory")
  
  set.seed(404)
  n <- 32
  p <- 5
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- runif(p, -1, 1)
  y <- drop(X %*% beta + rnorm(n, sd = 0.1))
  
  X_bm <- bigmemory::as.big.matrix(X)
  
  simpls_fit <- pls1_dense(X_bm, y, ncomp = 3, algorithm = "simpls")
  nipals_fit <- pls1_dense(X_bm, y, ncomp = 3, algorithm = "nipals")
  expect_equal(nipals_fit$ncomp, simpls_fit$ncomp)
  expect_equal(sort(names(nipals_fit)), sort(names(simpls_fit)))
  expect_equal(nipals_fit$coefficients, simpls_fit$coefficients, tolerance = 1e-6)
  expect_equal(nipals_fit$x_center, simpls_fit$x_center, tolerance = 1e-6)
  expect_equal(nipals_fit$x_scale, simpls_fit$x_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$y_center, simpls_fit$y_center, tolerance = 1e-6)
  expect_equal(nipals_fit$y_scale, simpls_fit$y_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$intercept, simpls_fit$intercept, tolerance = 1e-6)
  
  simpls_stream <- pls1_stream(X_bm, y, ncomp = 3, chunk_size = 6,
                                  algorithm = "simpls")
  nipals_stream <- pls1_stream(X_bm, y, ncomp = 3, chunk_size = 6,
                                  algorithm = "nipals")
  expect_equal(nipals_stream$ncomp, simpls_stream$ncomp)
  expect_equal(sort(names(nipals_stream)), sort(names(simpls_stream)))
  expect_equal(nipals_stream$coefficients, simpls_stream$coefficients, tolerance = 1e-6)
  expect_equal(nipals_stream$x_center, simpls_stream$x_center, tolerance = 1e-6)
  expect_equal(nipals_stream$x_scale, simpls_stream$x_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$y_center, simpls_stream$y_center, tolerance = 1e-6)
  expect_equal(nipals_stream$y_scale, simpls_stream$y_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$intercept, simpls_stream$intercept, tolerance = 1e-6)
})


test_that("big memory interfaces support both algorithms", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bigmemory")
  
  set.seed(404)
  n <- 32
  p <- 5
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- runif(p, -1, 1)
  y <- drop(X %*% beta + rnorm(n, sd = 0.1))
  
  X_bm <- bigmemory::as.big.matrix(X)
  
  simpls_fit <- pls1_dense_ya(X_bm, y, ncomp = 3, algorithm = "simpls")
  nipals_fit <- pls1_dense_ya(X_bm, y, ncomp = 3, algorithm = "nipals")
  expect_equal(nipals_fit$ncomp, simpls_fit$ncomp)
  expect_equal(sort(names(nipals_fit)), sort(names(simpls_fit)))
  expect_equal(nipals_fit$coefficients, simpls_fit$coefficients, tolerance = 1e-6)
  expect_equal(nipals_fit$x_center, simpls_fit$x_center, tolerance = 1e-6)
  expect_equal(nipals_fit$x_scale, simpls_fit$x_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$y_center, simpls_fit$y_center, tolerance = 1e-6)
  expect_equal(nipals_fit$y_scale, simpls_fit$y_scale, tolerance = 1e-6)
  expect_equal(nipals_fit$intercept, simpls_fit$intercept, tolerance = 1e-6)
  
  simpls_stream <- pls1_stream_ya(X_bm, y, ncomp = 3, chunk_size = 6,
                                          algorithm = "simpls")
  nipals_stream <- pls1_stream_ya(X_bm, y, ncomp = 3, chunk_size = 6,
                                          algorithm = "nipals")
  expect_equal(nipals_stream$ncomp, simpls_stream$ncomp)
  expect_equal(sort(names(nipals_stream)), sort(names(simpls_stream)))
  expect_equal(nipals_stream$coefficients, simpls_stream$coefficients, tolerance = 1e-6)
  expect_equal(nipals_stream$x_center, simpls_stream$x_center, tolerance = 1e-6)
  expect_equal(nipals_stream$x_scale, simpls_stream$x_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$y_center, simpls_stream$y_center, tolerance = 1e-6)
  expect_equal(nipals_stream$y_scale, simpls_stream$y_scale, tolerance = 1e-6)
  expect_equal(nipals_stream$intercept, simpls_stream$intercept, tolerance = 1e-6)
})
