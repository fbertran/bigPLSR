test_that("KF-PLS state API: batch parity and streamed near-parity", {
  skip_on_cran()
  set.seed(123)

  n <- 400; p <- 20; m <- 2; A <- 3
  X <- matrix(rnorm(n * p), n, p)
  # generate a linear-ish target with some noise
  Btrue <- matrix(c(0.8, -0.2,
                    0.0,  0.6,
                    rep(0, (p-2) * m)), nrow = p, byrow = TRUE)
  Y <- X %*% Btrue + matrix(rnorm(n * m, sd = 0.2), n, m)

  ## --- Reference: batch SIMPLS via pls_fit() --------------------------------
  fit_batch <- pls_fit(X, Y, ncomp = A, backend = "arma",
                       algorithm = "simpls", scores = "none", tol = 1e-10)
  # predictions
  Yhat_batch <- predict(fit_batch, X)

  ## --- Case 1: feed *all data in one* update; lambda = 1, q_proc = 0 --------
  st1 <- kf_pls_state_new(p, m, A, lambda = 1.0, q_proc = 0.0, r_meas = 0.0)
  kf_pls_state_update(st1, X, Y)   # single update with full data
  fit_state_1 <- kf_pls_state_fit(st1, tol = 1e-10)

  # sanity
  expect_s3_class(fit_state_1, "big_plsr")
  expect_true(all(c("coefficients","intercept","x_means","y_means","ncomp") %in% names(fit_state_1)))
  expect_equal(fit_state_1$ncomp, fit_batch$ncomp)

  # parity: coefficients and predictions should match very closely
  expect_equal(fit_state_1$coefficients, fit_batch$coefficients, tolerance = 1e-6)
  expect_equal(fit_state_1$intercept,     fit_batch$intercept,     tolerance = 1e-8)

  Yhat_state_1 <- predict(fit_state_1, X)
  expect_equal(Yhat_state_1, Yhat_batch, tolerance = 1e-6)

  ## --- Case 2: stream in two chunks; lambda ~ 1, tiny q_proc ----------------
  st2 <- kf_pls_state_new(p, m, A, lambda = 0.9999, q_proc = 1e-12, r_meas = 0.0)
  idx1 <- 1:(n/2); idx2 <- (n/2 + 1):n
  kf_pls_state_update(st2, X[idx1, , drop = FALSE], Y[idx1, , drop = FALSE])
  kf_pls_state_update(st2, X[idx2, , drop = FALSE], Y[idx2, , drop = FALSE])
  fit_state_2 <- kf_pls_state_fit(st2, tol = 1e-10)

  # near parity: predictions should correlate strongly with batch
  Yhat_state_2 <- predict(fit_state_2, X)

  # Use correlation across all predicted columns as a robust similarity metric
  cor1 <- cor(as.numeric(Yhat_state_2), as.numeric(Yhat_batch))
  expect_true(is.finite(cor1) && cor1 > 0.99)

  # Also check RMSE ratio isn’t crazy
  rmse <- function(a, b) sqrt(mean((a - b)^2))
  ratio <- rmse(Yhat_state_2, Y) / rmse(Yhat_batch, Y)
  expect_true(is.finite(ratio) && ratio < 1.05)  # within 5% of batch
})
