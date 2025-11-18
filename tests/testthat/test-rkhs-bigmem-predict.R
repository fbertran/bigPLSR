test_that("RKHS bigmem streamed predict matches dense predict", {
  skip_on_cran()
  skip_on_ci() 
  set.seed(123)
  n <- 80; p <- 6; m <- 2
  X  <- matrix(rnorm(n*p), n, p)
  Y  <- cbind(sin(X[,1]) + 0.5*X[,2]^2, cos(X[,3]) - 0.3*X[,4]^2) + matrix(rnorm(n*m, sd=0.02), n, m)
  Xbm <- bigmemory::as.big.matrix(X)
  Ybm <- bigmemory::as.big.matrix(Y)
  
  opts <- options(
    bigPLSR.kernel = "rbf",
    bigPLSR.rkhs.chunk_rows = 32L,
    bigPLSR.rkhs.chunk_cols = 32L
  ); on.exit(options(opts), add = TRUE)
  
  # Dense fit
  fit_d <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "rkhs",
                   tol = 1e-8, scores = "none")
  Yhat_d <- predict(fit_d, X)
  
  # Bigmem fit (stores X_ref + kstats in this patch)
  fit_b <- pls_fit(Xbm, Ybm, ncomp = 3, backend = "bigmem", algorithm = "rkhs",
                   tol = 1e-8, scores = "none")
  
  # Predict on training X using streamed path (pass descriptor explicitly)
  Yhat_b <- predict(fit_b, X, Xtrain = bigmemory::describe(Xbm))
  
  expect_equal(dim(Yhat_b), dim(Yhat_d))
  expect_equal(Yhat_b, Yhat_d, tolerance = 1e-6)
})