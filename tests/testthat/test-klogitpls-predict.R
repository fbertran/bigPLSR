test_that("klogitpls fits and predicts probabilities & scores (bigmem stream)", {
  skip_on_cran()
  skip_on_ci() 
  set.seed(7)
  n <- 120; p <- 8
  X  <- matrix(rnorm(n*p), n, p)
  eta <- 1 + 0.8*sin(X[,1]) - 0.6*X[,2]^2 + 0.3*X[,3]
  pr  <- 1/(1+exp(-eta))
  y   <- rbinom(n, 1, pr)
  
  Xbm <- bigmemory::as.big.matrix(X)
  ybm <- bigmemory::as.big.matrix(matrix(y, n, 1))
  
  opts <- options(
    bigPLSR.klogitpls.kernel = "rbf",
    bigPLSR.klogitpls.gamma  = 0.5,
    bigPLSR.klogitpls.degree = 3L,
    bigPLSR.klogitpls.coef0  = 1.0,
    bigPLSR.klogitpls.chunk_rows = 32L,
    bigPLSR.klogitpls.chunk_cols = 32L
  ); on.exit(options(opts), add = TRUE)
  
  # Dense baseline
  fit_d <- pls_fit(X, y, ncomp = 2, backend = "arma", algorithm = "klogitpls",
                   tol = 1e-8, scores = "none",
                   class_weights = c("0" = 1, "1" = 1))
  
  p_d   <- predict(fit_d, X, type = "response")
  T_d   <- predict(fit_d, X, type = "scores")
  expect_true(all(p_d >= 0 & p_d <= 1))
  expect_equal(ncol(T_d), fit_d$ncomp)
  
  # Bigmem fit (stores X_ref + kstats)
  fit_b <- pls_fit(Xbm, ybm, ncomp = 2, backend = "bigmem", algorithm = "klogitpls",
                   tol = 1e-8, scores = "none",
                   class_weights = c("0" = 1, "1" = 1))
  
  # probabilities via streamed cross-kernel
  p_b <- predict(fit_b, X, Xtrain = bigmemory::describe(Xbm))
  expect_true(all(p_b >= 0 & p_b <= 1))
  expect_equal(length(p_b), n)
  
  # scores via streamed cross-kernel
  T_b <- predict(fit_b, X, type = "scores", Xtrain = bigmemory::describe(Xbm))
  expect_equal(nrow(T_b), n)
  expect_equal(ncol(T_b), fit_b$ncomp)
  
  # parity with dense (allow small numeric differences)
  expect_equal(p_b, as.numeric(p_d), tolerance = 1e-6)
})
