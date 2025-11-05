
test_that("dense and bigmem backends agree on small data", {
  set.seed(1)
  n <- 200; p <- 30
  X <- matrix(rnorm(n*p), n, p)
  y <- X[,1]*2 - X[,2]*1 + rnorm(n)

  fit_dense <- pls_fit(X, y, ncomp = 2, backend = "arma", scores = "r")
  # Create file-backed big.matrix
  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))

  fit_big <- pls_fit(bmX, bmy, ncomp = 2, backend = "bigmem", scores = "none")

  expect_equal(length(fit_dense$coefficients), p)
  expect_equal(length(fit_big$coefficients), p)
  cor_coefs <- cor(as.numeric(fit_dense$coefficients), as.numeric(fit_big$coefficients))
  expect_gt(cor_coefs, 0.95)
})

test_that("scores='big' returns a big.matrix of correct dimension", {
  set.seed(2)
  n <- 150; p <- 20
  X <- matrix(rnorm(n*p), n, p)
  y <- X[,1]*0.5 + rnorm(n)

  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))

  fit <- pls_fit(bmX, bmy, ncomp = 3, backend = "bigmem", scores = "big")
  expect_true(inherits(fit$scores, "big.matrix"))
  expect_equal(nrow(fit$scores), n)
  expect_equal(ncol(fit$scores), fit$ncomp)
})
