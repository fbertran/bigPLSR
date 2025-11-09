test_that("RKHS predict (rbf, dense) matches centered cross-kernel formula", {
  skip_if_not_installed("bigPLSR")  # if testing from another pkg
  set.seed(42)
  n <- 40; p <- 6; m <- 2
  X <- matrix(rnorm(n*p), n, p)
  Y <- X[,1:2,drop=FALSE] %*% matrix(c(1,-2, 0.5,1), 2, m) + matrix(rnorm(n*m, sd=0.1), n, m)
  
  Xtr <- X[1:30,]; Ytr <- Y[1:30,]
  Xte <- X[31:40,]
  
  gamma <- 1/p
  fit <- pls_fit(Xtr, Ytr, ncomp=3, backend="arma",
                 algorithm="rkhs", kernel="rbf",
                 gamma=gamma, degree=3L, coef0=1,
                 scores="none")
  
  pred <- predict(fit, Xte)
  
  # manual reference: K* centered with training stats
  # 1) Build the (uncentered) cross-kernel between test and train:
  K_te_tr <- bigPLSR:::.bigPLSR_make_kernel(
    Xte, fit$X,                               # needs training X if available
    kernel = fit$kernel %||% fit$kernel_x %||% "rbf",
    gamma  = fit$gamma  %||% fit$gamma_x  %||% (1 / ncol(Xtr)),
    degree = fit$degree %||% fit$degree_x %||% 3L,
    coef0  = fit$coef0  %||% fit$coef0_x  %||% 1
  )
  
  # 2) Obtain training kernel centering stats:
  #   m_train = colMeans(K_train,train), mu = mean(K_train,train)
  if (!is.null(fit$k_colmeans) && !is.null(fit$k_grandmean)) {
    m_train <- fit$k_colmeans
    mu      <- fit$k_grandmean
  } else {
    K_tr_tr <- bigPLSR:::.bigPLSR_make_kernel(
      fit$X, fit$X,
      kernel = fit$kernel %||% fit$kernel_x %||% "rbf",
      gamma  = fit$gamma  %||% fit$gamma_x  %||% (1 / ncol(Xtr)),
      degree = fit$degree %||% fit$degree_x %||% 3L,
      coef0  = fit$coef0  %||% fit$coef0_x  %||% 1
    )
    m_train <- colMeans(K_tr_tr)
    mu      <- mean(K_tr_tr)
  }
  
  # 3) Center the cross-kernel with training stats (H_te K H_tr):
  m_test <- rowMeans(K_te_tr)
  Kc_te  <- sweep(K_te_tr, 2, m_train, "-")
  Kc_te  <- sweep(Kc_te,   1, m_test,  "-")
  Kc_te  <- Kc_te + mu
  
  # 4) Predict with dual coefficients (alpha) and add back y means:
  alpha  <- fit$dual_coef %||% fit$coefficients  # both supported in patch below
  Yhat_c <- Kc_te %*% alpha
  Yhat   <- sweep(Yhat_c, 2, fit$y_means, "+")
  
  expect_equal(pred, Yhat, tolerance = 1e-7)
})
