test_that("KF-PLS (dense) approximates SIMPLS when lambda≈1", {
  skip_on_cran()
  set.seed(123)
  n <- 300; p <- 40; m <- 2
  X <- matrix(rnorm(n*p), n, p)
  beta <- rbind(c(0.7,  -0.2),
                c(-0.4,  0.5),
                matrix(0, p-2, m))
  Y <- X %*% beta + matrix(rnorm(n*m, sd=0.1), n, m)
  
  fit_s <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "simpls", scores = "none")
  expect_s3_class(fit_s, "big_plsr")
  
  old <- options(bigPLSR.kf.lambda = 0.999, bigPLSR.kf.q_proc = 1e-8)
  on.exit(options(old), add=TRUE)
  fit_kf <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "kf_pls", scores = "none")
  expect_s3_class(fit_kf, "big_plsr")
  
  # coefficients & preds should be close
  expect_equal(fit_kf$coefficients, fit_s$coefficients, tolerance = 5e-2)
  pr_s  <- predict(fit_s, X)
  pr_kf <- predict(fit_kf, X)
  expect_equal(pr_kf, pr_s, tolerance = 6e-2)
})

test_that("KF-PLS bigmem ~ dense parity (predictions)", {
  skip_on_cran()
  set.seed(124)
  n <- 200; p <- 30
  X <- matrix(rnorm(n*p), n, p)
  y <- X[,1]*0.8 - X[,2]*0.5 + rnorm(n, sd=0.2)
  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))
  
  old <- options(bigPLSR.kf.lambda = 0.995, bigPLSR.kf.q_proc = 1e-6)
  on.exit(options(old), add=TRUE)
  
  fit_d <- pls_fit(X, y, ncomp = 2, backend = "arma", algorithm = "kf_pls", scores = "r")
  fit_b <- pls_fit(bmX, bmy, ncomp = 2, backend = "bigmem", algorithm = "kf_pls",
                   scores = "r", chunk_size = 64L)
  pr_d <- predict(fit_d, X)
  pr_b <- predict(fit_b, X)
  expect_equal(drop(pr_b), drop(pr_d), tolerance = 8e-2)
  # scores exist & have expected dimensions
  expect_true(ncol(predict(fit_b, X, type="scores")) == fit_b$ncomp)
})
