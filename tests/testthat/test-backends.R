
make_filebacked_big_matrix <- function(x, prefix) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  id <- paste(prefix, Sys.getpid(), sample.int(100000000L, 1L), sep = "_")
  backingfile <- paste0(id, ".bin")
  descriptorfile <- paste0(id, ".desc")
  files <- file.path(tempdir(), c(backingfile, descriptorfile))
  unlink(files)
  bm <- bigmemory::filebacked.big.matrix(
    nrow = nrow(x), ncol = ncol(x), type = "double",
    backingfile = backingfile,
    backingpath = tempdir(),
    descriptorfile = descriptorfile
  )
  bm[,] <- x
  list(bm = bm, files = files)
}

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
  
  tmp <- tempdir()
  if(file.exists(paste(tmp,"scores.desc",sep="/"))){unlink(paste(tmp,"scores.desc",sep="/"))}
  if(file.exists(paste(tmp,"scores.bin",sep="/"))){unlink(paste(tmp,"scores.bin",sep="/"))}
  
  fit <- pls_fit(bmX, bmy, ncomp = 3, backend = "bigmem", scores = "big", 
                 scores_backingfile="scores.bin", scores_backingpath=tmp, 
                 scores_descriptorfile="scores.desc")
  expect_true(inherits(fit$scores, "big.matrix"))
  expect_equal(nrow(fit$scores), n)
  expect_equal(ncol(fit$scores), fit$ncomp)
})

test_that("bigmem NIPALS PLS1 routes through wide-safe streaming backend", {
  set.seed(101)
  n <- 60
  p <- 25
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(1.5 * X[, 1] - 0.7 * X[, 2] + rnorm(n, sd = 0.1), n, 1)

  x_fb <- make_filebacked_big_matrix(X, "pls1_small_X")
  y_fb <- make_filebacked_big_matrix(Y, "pls1_small_Y")
  on.exit(unlink(c(x_fb$files, y_fb$files)), add = TRUE)

  fit <- pls_fit(
    x_fb$bm, y_fb$bm,
    ncomp = 2,
    backend = "bigmem",
    mode = "pls1",
    algorithm = "nipals",
    scores = "none",
    chunk_size = 128
  )

  coef <- as.matrix(fit$coefficients)
  expect_equal(fit$mode, "pls1")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(nrow(coef), p)
  expect_equal(ncol(coef), 1L)
  expect_gt(fit$ncomp, 0)
  expect_lte(fit$ncomp, 2)
  expect_null(fit$scores)
})

test_that("bigmem NIPALS PLS1 handles moderately wide predictors", {
  set.seed(102)
  n <- 30
  p <- 3000
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(0.8 * X[, 1] - 0.3 * X[, 17] + rnorm(n, sd = 0.2), n, 1)

  x_fb <- make_filebacked_big_matrix(X, "pls1_wide_X")
  y_fb <- make_filebacked_big_matrix(Y, "pls1_wide_Y")
  on.exit(unlink(c(x_fb$files, y_fb$files)), add = TRUE)

  fit <- pls_fit(
    x_fb$bm, y_fb$bm,
    ncomp = 2,
    backend = "bigmem",
    mode = "pls1",
    algorithm = "nipals",
    scores = "none",
    chunk_size = 16
  )

  coef <- as.matrix(fit$coefficients)
  expect_equal(fit$mode, "pls1")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(dim(coef), c(p, 1L))
  expect_null(fit$scores)
})

test_that("bigmem NIPALS PLS2 routing remains available", {
  set.seed(103)
  n <- 50
  p <- 18
  q <- 3
  X <- matrix(rnorm(n * p), n, p)
  B <- matrix(rnorm(p * q), p, q)
  Y <- X %*% B + matrix(rnorm(n * q, sd = 0.1), n, q)

  x_fb <- make_filebacked_big_matrix(X, "pls2_X")
  y_fb <- make_filebacked_big_matrix(Y, "pls2_Y")
  on.exit(unlink(c(x_fb$files, y_fb$files)), add = TRUE)

  fit <- pls_fit(
    x_fb$bm, y_fb$bm,
    ncomp = 2,
    backend = "bigmem",
    mode = "pls2",
    algorithm = "nipals",
    scores = "none",
    chunk_size = 64
  )

  expect_equal(fit$mode, "pls2")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(dim(as.matrix(fit$coefficients)), c(p, q))
  expect_null(fit$scores)
})

test_that("explicit bigmem NIPALS PLS2 mode is preserved for one-column response", {
  set.seed(104)
  n <- 50
  p <- 18
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(X[, 1] + 0.5 * X[, 3] + rnorm(n, sd = 0.1), n, 1)

  x_fb <- make_filebacked_big_matrix(X, "pls2_one_col_X")
  y_fb <- make_filebacked_big_matrix(Y, "pls2_one_col_Y")
  on.exit(unlink(c(x_fb$files, y_fb$files)), add = TRUE)

  fit <- pls_fit(
    x_fb$bm, y_fb$bm,
    ncomp = 2,
    backend = "bigmem",
    mode = "pls2",
    algorithm = "nipals",
    scores = "none",
    chunk_size = 64
  )

  expect_equal(fit$mode, "pls2")
  expect_equal(fit$algorithm, "nipals")
  expect_equal(dim(as.matrix(fit$coefficients)), c(p, 1L))
  expect_null(fit$scores)
})
