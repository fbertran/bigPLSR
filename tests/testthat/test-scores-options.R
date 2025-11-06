test_that("dense: scores_colnames works; descriptor absent", {
  skip_on_cran()
  set.seed(42)
  
  n <- 120; p <- 20; k <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- X[,1]*0.7 - X[,2]*0.3 + rnorm(n, 0.1)
  
  nm <- paste0("t", seq_len(k))
  fit <- pls_fit(X, y, ncomp = k,
                 backend = "arma", scores = "r",
                 scores_colnames = nm,
                 return_scores_descriptor = TRUE)
  
  expect_true(is.matrix(fit$scores))
  expect_equal(ncol(fit$scores), k)
  expect_equal(colnames(fit$scores), nm)
  # Dense path shouldn't add a descriptor:
  expect_false("scores_descriptor" %in% names(fit))
})

test_that("bigmem: file-backed sink → named columns + descriptor", {
  skip_on_cran()
  skip_if_not_installed("bigmemory")
  options_val_before <- options("bigmemory.allow.dimnames")
  options(bigmemory.allow.dimnames=TRUE)
  set.seed(43)
  
  n <- 150; p <- 25; k <- 2
  X <- matrix(rnorm(n * p), n, p)
  y <- X[,1]*0.5 + rnorm(n, 0.2)
  
  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))
  
  tmp <- tempdir()
  if(file.exists(paste(tmp,"scores.desc",sep="/"))){unlink(paste(tmp,"scores.desc",sep="/"))}
  if(file.exists(paste(tmp,"scores.bin",sep="/"))){unlink(paste(tmp,"scores.bin",sep="/"))}
  sink_bm <- bigmemory::filebacked.big.matrix(
    nrow = n, ncol = k, type = "double",
    backingfile = "scores.bin", backingpath = tmp,
    descriptorfile = "scores.desc"
  )
  
  nm <- paste0("comp", seq_len(k))
  fit <- pls_fit(bmX, bmy, ncomp = k,
                 backend = "bigmem", scores = "big",
                 scores_target = "existing", scores_bm = sink_bm,
                 scores_colnames = nm,
                 return_scores_descriptor = TRUE)
  
  expect_true(inherits(fit$scores, "big.matrix"))
  expect_equal(nrow(fit$scores), n)
  expect_equal(ncol(fit$scores), k)
  expect_equal(colnames(fit$scores), nm)
  
  # Descriptor is returned and attachable:
  expect_true("scores_descriptor" %in% names(fit))
  desc <- fit$scores_descriptor
  expect_true(inherits(desc, "big.matrix.descriptor"))
  # quick attach smoke test:
  attached <- bigmemory::attach.big.matrix(desc)
  expect_true(inherits(attached, "big.matrix"))
  options(bigmemory.allow.dimnames=options_val_before)
})

test_that("bigmem: descriptor sink works with names + descriptor", {
  skip_on_cran()
  skip_if_not_installed("bigmemory")
  options_val_before <- options("bigmemory.allow.dimnames")
  options(bigmemory.allow.dimnames=TRUE)
  set.seed(44)
  
  n <- 80; p <- 15; k <- 2
  X <- matrix(rnorm(n * p), n, p)
  y <- X[,1] - 0.2*X[,2] + rnorm(n, 0.05)
  
  bmX <- bigmemory::as.big.matrix(X)
  bmy <- bigmemory::as.big.matrix(matrix(y, n, 1))
  
  tmp <- tempdir()
  sink_bm <- bigmemory::filebacked.big.matrix(
    nrow = n, ncol = k, type = "double",
    backingfile = "scores2.bin", backingpath = tmp,
    descriptorfile = "scores2.desc"
  )
  sink_desc <- bigmemory::describe(sink_bm)
  
  nm <- c("t1","t2")
  fit <- pls_fit(bmX, bmy, ncomp = k,
                 backend = "bigmem", scores = "big",
                 scores_target = "existing", scores_bm = sink_desc,
                 scores_colnames = nm,
                 return_scores_descriptor = TRUE)
  
  expect_true(inherits(fit$scores, "big.matrix"))
  expect_equal(colnames(fit$scores), nm)
  expect_true("scores_descriptor" %in% names(fit))
  expect_true(inherits(fit$scores_descriptor, "big.matrix.descriptor"))
  options(bigmemory.allow.dimnames=options_val_before)
})
