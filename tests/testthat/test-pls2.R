test_that("PLS2 bigmem path matches dense fallback on small data", {
  skip_on_cran(); skip_on_ci(); skip_if_not_installed("bigmemory")
  options_val_before <- options("bigmemory.allow.dimnames")
  options(bigmemory.allow.dimnames=TRUE)
  set.seed(1)
  n <- 200; p <- 15; m <- 3; k <- 2
  X <- matrix(rnorm(n*p), n, p)
  B <- matrix(rnorm(p*m), p, m)
  Y <- X %*% B + matrix(rnorm(n*m, sd=0.05), n, m)
  
  # Dense reference (you can call your arma+pls2 fallback)
  ref <- pls_fit(X, Y, ncomp=k, backend="arma", mode="pls2", scores="none")
  
  bmX <- bigmemory::as.big.matrix(X)
  bmY <- bigmemory::as.big.matrix(Y)
  
  fit <- pls_fit(bmX, bmY, ncomp=k, backend="bigmem", mode="pls2", scores="none")
  
  expect_equal(dim(fit$coefficients), dim(ref$coefficients))
  expect_equal(as.numeric(fit$coefficients), as.numeric(ref$coefficients), tolerance=1e-6)
  expect_equal(as.numeric(fit$intercept),   as.numeric(ref$intercept),   tolerance=1e-6)
  options(bigmemory.allow.dimnames=options_val_before)
})

test_that("PLS2 bigmem streams scores correctly", {
  skip_on_cran(); skip_on_ci(); skip_if_not_installed("bigmemory")
  options_val_before <- options("bigmemory.allow.dimnames")
  options(bigmemory.allow.dimnames=TRUE)
  set.seed(2)
  n <- 150; p <- 12; m <- 2; k <- 2
  X <- matrix(rnorm(n*p), n, p)
  B <- matrix(rnorm(p*m), p, m)
  Y <- scale(X, scale=FALSE) %*% B + matrix(rnorm(n*m, sd=0.1), n, m)
  
  bmX <- bigmemory::as.big.matrix(X)
  bmY <- bigmemory::as.big.matrix(Y)
  
  tmp <- tempdir()
  sink <- bigmemory::filebacked.big.matrix(nrow=n, ncol=k, type="double",
                                           backingfile="scores_pls2.bin",
                                           backingpath=tmp,
                                           descriptorfile="scores_pls2.desc")
  fit <- pls_fit(bmX, bmY, ncomp=k, backend="bigmem", mode="pls2",
                 scores="big", scores_target="existing", scores_bm=sink,
                 scores_colnames=paste0("t",1:k), return_scores_descriptor=TRUE)
  expect_true(inherits(fit$scores, "big.matrix"))
  expect_equal(nrow(fit$scores), n); expect_equal(ncol(fit$scores), k)
  expect_equal(colnames(fit$scores), paste0("t",1:k))
  expect_true("scores_descriptor" %in% names(fit))
  options(bigmemory.allow.dimnames=options_val_before)
})
