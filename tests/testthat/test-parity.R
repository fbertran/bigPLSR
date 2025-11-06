
test_that("dense vs bigmem parity is tight under deterministic settings", {
  skip_on_cran()
  set.seed(123)
  n <- 200; p <- 40
  X <- matrix(rnorm(n*p), n, p)
  y <- X[,1]*2 - X[,2]*0.5 + rnorm(n, sd = 0.1)

  fit_dense <- pls_fit(X, y, ncomp = 3, tol = 1e-10, backend = "arma", scores = "r")

  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))
  fit_big  <- pls_fit(bmX, bmy, ncomp = 3, tol = 1e-10, backend = "bigmem", scores = "none")

  expect_equal(as.numeric(fit_dense$coefficients), as.numeric(fit_big$coefficients), tolerance = 1e-7)
  expect_equal(as.numeric(fit_dense$intercept), as.numeric(fit_big$intercept), tolerance = 1e-7)
})

test_that("file-backed scores sink works and matches dense scores on small data", {
  skip_on_cran()
  set.seed(321)
  n <- 120; p <- 25; k <- 2
  X <- matrix(rnorm(n*p), n, p)
  y <- X[,1]*0.8 + rnorm(n, sd = 0.2)

  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))

  fit_dense <- pls_fit(X, y, ncomp = k, backend = "arma", scores = "r")

  tmp <- tempdir()
  sink_bm <- bigmemory::filebacked.big.matrix(nrow=n, ncol=k, type="double",
                                              backingfile="scores.bin", backingpath=tmp,
                                              descriptorfile="scores.desc")
  fit_big <- pls_fit(bmX, bmy, ncomp = k, backend = "bigmem", scores = "big",
                     scores_target = "existing", scores_bm = sink_bm)

  expect_true(inherits(fit_big$scores, "big.matrix"))
  expect_equal(nrow(fit_big$scores), n)
  expect_equal(ncol(fit_big$scores), k)

  scores_from_file <- as.matrix(fit_big$scores[])
  expect_equal(scores_from_file, fit_dense$scores, tolerance = 1e-6)
})
