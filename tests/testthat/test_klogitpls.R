# tests/testthat/test-klogitpls.R
test_that("klogitpls fits and predicts probabilities with class weights", {
  skip_on_cran()
  set.seed(123)
  
  n <- 200; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  
  # Nonlinear signal (radial threshold in first two coords) + noise
  r2 <- rowSums(X[, 1:2]^2)
  y  <- as.integer(r2 + 0.20 * rnorm(n) > 1.5)
  
  # Unweighted klogitpls (uses KPLS + IRLS in latent space)
  fit <- pls_fit(
    X, y,
    ncomp     = 2,
    backend   = "arma",
    algorithm = "klogitpls",
    kernel    = "rbf",
    gamma     = 1.0 / p,
    scores    = "none"
  )
  
  expect_s3_class(fit, "big_plsr")
  expect_identical(tolower(fit$algorithm), "klogitpls")
  
  pr <- predict(fit, X)
  pr <- as.numeric(pr)
  expect_equal(length(pr), n)
  expect_true(all(is.finite(pr)))
  expect_true(all(pr >= 0 & pr <= 1))
  
  # Weighted fit: upweight the positive (minority) class
  w_pos <- 2.0
  w_neg <- 0.5
  fit_w <- pls_fit(
    X, y,
    ncomp         = 2,
    backend       = "arma",
    algorithm     = "klogitpls",
    kernel        = "rbf",
    gamma         = 1.0 / p,
    class_weights = c(w_neg, w_pos),  # c(weight for 0, weight for 1)
    scores        = "none"
  )
  
  pr_w <- predict(fit_w, X)
  expect_equal(length(pr_w), n)
  expect_true(all(pr_w >= 0 & pr_w <= 1))
  
  # Class-weighting should increase the mean probability on the positive class
  m_pos_unw <- mean(pr[y == 1])
  m_pos_w   <- mean(pr_w[y == 1])
  expect_gt(m_pos_w, m_pos_unw - 1e-8)  # strictly greater (with tiny slack)
  
  # And typically should not inflate negatives (allowing tiny numerical wiggle)
  m_neg_unw <- mean(pr[y == 0])
  m_neg_w   <- mean(pr_w[y == 0])
  expect_lte(m_neg_w, m_neg_unw + 1e-6)
})


test_that("klogitpls (dense RBF) learns a non-linear boundary", {
  skip_on_cran()
  set.seed(123)
  n <- 200; p <- 5
  X <- matrix(rnorm(n*p), n, p)
  f <- sin(X[,1]) + 0.3*X[,2]^2 - 0.2*X[,3]
  y <- as.integer(f + rnorm(n, sd=0.2) > 0)
  # train/test split
  id <- sample.int(n, n %/% 2)
  Xtr <- X[id,]; Xte <- X[-id,]
  ytr <- y[id];   yte <- y[-id]
  # fit rkhslogitpls
  opts <- options(
    bigPLSR.klogitpls.kernel="rbf", bigPLSR.klogitpls.gamma=0.7,
    bigPLSR.klogitpls.degree=3L,    bigPLSR.klogitpls.coef0=1.0,
    bigPLSR.klogitpls.max_irls_iter=50L, bigPLSR.klogitpls.alt_updates=1L
  ); on.exit(options(opts), add=TRUE)
  fit <- pls_fit(Xtr, ytr, ncomp=3, backend="arma", algorithm="klogitpls")
  pr  <- predict(fit, Xte)
  p1  <- pr
  auc_like <- cor(p1, yte) # cheap monotone metric
  expect_true(!is.na(auc_like) && auc_like > 0.5)
})

