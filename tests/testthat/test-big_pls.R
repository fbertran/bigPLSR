test_that("big_pls_fit approximates OLS when enough components are used", {
  skip_on_cran()
  skip_if_not_installed("bigmemory")
  
  set.seed(123)
  n <- 30
  p <- 5
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- runif(p, -1, 1)
  y <- as.numeric(X %*% beta + rnorm(n, sd = 0.01))
  
  bm <- bigmemory::as.big.matrix(X)
  fit <- big_pls_fit(bm, y, ncomp = p, center = TRUE, scale = FALSE,
                     center_y = TRUE, scale_y = FALSE)
  
  expect_equal(length(fit$coefficients), p)
  preds <- drop(cbind(1, X) %*% c(fit$intercept, fit$coefficients))
  expect_lt(max(abs(preds - y)), 0.25)
})

test_that("streaming fit matches direct fit", {
  skip_on_cran()
  skip_if_not_installed("bigmemory")
  
  set.seed(321)
  n <- 40
  p <- 6
  X <- matrix(rnorm(n * p), nrow = n)
  beta <- rnorm(p)
  y <- as.numeric(X %*% beta + rnorm(n, sd = 0.05))
  
  bm <- bigmemory::as.big.matrix(X)
  direct <- big_pls_fit(bm, y, ncomp = 4, center = TRUE, scale = TRUE,
                        center_y = TRUE, scale_y = TRUE)
  streaming <- big_pls_stream_fit(bm, y, ncomp = 4, chunk_size = 7,
                                  center = TRUE, scale = TRUE,
                                  center_y = TRUE, scale_y = TRUE)
  
  expect_equal(streaming$ncomp, direct$ncomp)
  expect_equal(streaming$intercept, direct$intercept, tolerance = 1e-8)
  expect_equal(streaming$coefficients, direct$coefficients, tolerance = 1e-8)
})
