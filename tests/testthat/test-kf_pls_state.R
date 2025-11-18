test_that("KF-PLS state API: batch parity and streamed near-parity", {
  skip_on_cran()
  skip_on_ci() 
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


test_that("KF-PLS state API tracks full SIMPLS on concatenated data", {
  skip_on_cran()
  skip_on_ci() 
  set.seed(123)
  n <- 200; p <- 15; m <- 3; A <- 4
  X <- matrix(rnorm(n*p), n, p)
  B <- matrix(rnorm(p*m), p, m)
  Y <- scale(X, TRUE, FALSE) %*% B + 0.1*matrix(rnorm(n*m), n, m)
  
  # split stream
  idx <- sample.int(n)
  X1 <- X[idx[1:100], , drop=FALSE]
  Y1 <- Y[idx[1:100], , drop=FALSE]
  X2 <- X[idx[101:200], , drop=FALSE]
  Y2 <- Y[idx[101:200], , drop=FALSE]
  
  # stateful KF-PLS with mild forgetting
  st <- cpp_kf_pls_state_new(p, m, A, lambda = 0.98, q_proc = 1e-6, r_meas = 0.0)
  cpp_kf_pls_state_update(st, X1, Y1)
  cpp_kf_pls_state_update(st, X2, Y2)
  fit_state <- cpp_kf_pls_state_fit(st, tol = 1e-8)
  
  # full SIMPLS on all data via high-level API
  fit_full <- bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                               algorithm = "simpls", scores = "none")
  
  expect_type(fit_state, "list") # from C++; R wrapper may .finalize_pls_fit
  B1 <- as.matrix(fit_state$coefficients)
  B2 <- as.matrix(fit_full$coefficients)
  # compare up to a mild tolerance (forgetting & ridge can shift magnitude slightly)
  expect_lt(norm(B1 - B2, "F") / (1e-12 + norm(B2, "F")), 0.10)
  
  # prediction parity on held-out slice of X
  Xh <- matrix(rnorm(50*p), 50, p)
  ph <- function(fit) {
    pr <- predict(bigPLSR:::.finalize_pls_fit(fit, "kf_pls"), Xh)
    as.matrix(pr)
  }
  Yp_state <- ph(fit_state)
  Yp_full  <- predict(fit_full, Xh)
  expect_lt(mean((Yp_state - Yp_full)^2), 1e-1)
})

test_that("KF-PLS state: exact parity with lambda=1, q_proc=0", {
  set.seed(1); n <- 200; p <- 15; m <- 3; A <- 4
  X <- matrix(rnorm(n*p), n, p)
  B <- matrix(rnorm(p*m), p, m)
  Y <- scale(X, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n*m), n, m)
  X1 <- X[1:100,]; Y1 <- Y[1:100,]
  X2 <- X[101:200,]; Y2 <- Y[101:200,]
  st <- cpp_kf_pls_state_new(p, m, A, lambda = 1.0, q_proc = 0.0, r_meas = 0.0)
  cpp_kf_pls_state_update(st, X1, Y1)
  cpp_kf_pls_state_update(st, X2, Y2)
  fit_state <- cpp_kf_pls_state_fit(st, tol = 1e-8)
  fit_full  <- bigPLSR::pls_fit(X, Y, ncomp=A, backend="arma", algorithm="simpls", scores="none")
  Yp_state  <- predict(bigPLSR:::.finalize_pls_fit(fit_state, "kf_pls"), X)
  Yp_full   <- predict(fit_full, X)
  expect_lt(mean((Yp_state - Yp_full)^2), 1e-8)
})

test_that("KF-PLS state: near-parity with forgetting (lambda=0.98)", {
  set.seed(2); n <- 200; p <- 15; m <- 3; A <- 4
  X <- matrix(rnorm(n*p), n, p)
  B <- matrix(rnorm(p*m), p, m)
  Y <- scale(X, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n*m), n, m)
  X1 <- X[1:100,]; Y1 <- Y[1:100,]
  X2 <- X[101:200,]; Y2 <- Y[101:200,]
  st <- cpp_kf_pls_state_new(p, m, A, lambda = 0.98, q_proc = 1e-6, r_meas = 0.0)
  cpp_kf_pls_state_update(st, X1, Y1)
  cpp_kf_pls_state_update(st, X2, Y2)
  fit_state <- cpp_kf_pls_state_fit(st, tol = 1e-8)
  fit_full  <- bigPLSR::pls_fit(X, Y, ncomp=A, backend="arma", algorithm="simpls", scores="none")
  Yp_state  <- predict(bigPLSR:::.finalize_pls_fit(fit_state, "kf_pls"), X)
  Yp_full   <- predict(fit_full, X)
  # looser bound (empirically ~0.05–0.09 depending on RNG)
  expect_lt(mean((Yp_state - Yp_full)^2), 0.1)
})
