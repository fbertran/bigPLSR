test_that("rkhs_xy: predict parity on dense RBF example + orthonormal scores", {
  skip_on_cran()
  set.seed(42)
  
  n <- 60; p <- 6; m <- 2
  X <- matrix(rnorm(n * p), n, p)
  # non-linear Y tied to X
  f1 <- sin(X[,1]) + 0.4 * X[,2]^2
  f2 <- cos(X[,3]) - 0.3 * X[,4]^2
  Y  <- cbind(f1, f2) + matrix(rnorm(n * m, sd = 0.05), n, m)
  
  # Kernel & ridge settings (make them explicit so the test is stable)
  
  opts_old <- options(
    bigPLSR.rkhs_xy.kernel_x = "rbf",
    bigPLSR.rkhs_xy.gamma_x  = 0.5,     # gamma for RBF(X,X)
    bigPLSR.rkhs_xy.degree_x = 3L,
    bigPLSR.rkhs_xy.coef0_x  = 1.0,
    bigPLSR.rkhs_xy.kernel_y = "linear",
    bigPLSR.rkhs_xy.gamma_y  = 1.0,     # unused for linear
    bigPLSR.rkhs_xy.degree_y = 3L,
    bigPLSR.rkhs_xy.coef0_y  = 1.0,
    bigPLSR.rkhs_xy.lambda_x = 1e-6,
    bigPLSR.rkhs_xy.lambda_y = 1e-6,
    bigPLSR.store_X_max      = 10000L
  )
  on.exit(options(opts_old), add = TRUE)
  
  fit <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "rkhs_xy", tol = 1e-8)
  
  # basic structure
  expect_s3_class(fit, "big_plsr")
  expect_true(all(c("dual_coef", "scores", "intercept", "ncomp") %in% names(fit)))
  expect_equal(nrow(fit$dual_coef), n)
  expect_equal(ncol(fit$dual_coef), m)
  expect_true(fit$ncomp >= 1)
  
  # helper: RBF kernel + double-centering wrt train
  rbf <- function(A, B, gamma) {
    a2 <- rowSums(A*A); b2 <- rowSums(B*B)
    D2 <- outer(a2, b2, "+") - 2 * (A %*% t(B))
    exp(-gamma * D2)
  }
  H_center_cross <- function(K_cross, K_train) {
    # K_train must be the uncentered Gram(X,X)
    n <- nrow(K_train)
    one_n <- matrix(1/n, n, n)
    colm <- colMeans(K_train)          # 1 x n
    mu   <- mean(K_train)              # scalar
    # Center cross-kernel (rows = X*, cols = Xtrain)
    rowm_cross <- rowMeans(K_cross)    # n* x 1
    K_cross - outer(rowm_cross, rep(1, n)) - 
      matrix(rep(colm, each = nrow(K_cross)), nrow(K_cross), n, byrow = FALSE) + mu
  }
  
  # Manual “train” prediction using the *same* centering stats used in training
  # (pull from the model; fall back to options only for kernel hyperparams)
  kernel_x <- fit$kernel_x %||% getOption("bigPLSR.rkhs_xy.kernel_x", "rbf")
  gamma_x  <- fit$gamma_x  %||% getOption("bigPLSR.rkhs_xy.gamma_x",  1 / ncol(X))
  degree_x <- fit$degree_x %||% getOption("bigPLSR.rkhs_xy.degree_x", 3L)
  coef0_x  <- fit$coef0_x  %||% getOption("bigPLSR.rkhs_xy.coef0_x",  1.0)
  
  # Build *uncentered* train Gram with exactly the same kernel hyperparams
  K_train <- bigPLSR:::.bigPLSR_make_kernel(X, X, kernel_x, gamma_x, degree_x, coef0_x)
  
  # Use the training centering stats saved by the trainer
  kstats <- fit$kstats_x %||% list(r = fit$kx_colmeans %||% fit$k_colmeans,
                                   g = fit$kx_grandmean %||% fit$k_grandmean)
  stopifnot(!is.null(kstats$r), !is.null(kstats$g))
  
  # Double-center the train Gram with the model’s statistics
  Kc_train <- .bigPLSR_center_cross_kernel(K_train, r_train = kstats$r, g_train = kstats$g)
  
  Yhat_train_manual <- Kc_train %*% fit$dual_coef +
    matrix(fit$intercept, n, m, byrow = TRUE)
  
  # predict() on training X should match manual formula closely
  Yhat_train_pred <- predict(fit, X)
  expect_equal(Yhat_train_pred, Yhat_train_manual, tolerance = 1e-6)
  
  # Scores orthonormality (approx): T' T ~ I
  G <- crossprod(fit$scores)
  I <- diag(ncol(G))
  expect_lt(max(abs(G - I)), 1e-5)
})
