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
  # fit klogitpls
  opts <- options(
    bigPLSR.klogitpls.kernel="rbf", bigPLSR.klogitpls.gamma=0.7,
    bigPLSR.klogitpls.degree=3L,    bigPLSR.klogitpls.coef0=1.0,
    bigPLSR.klogitpls.max_irls_iter=50L, bigPLSR.klogitpls.alt_updates=1L
  ); on.exit(options(opts), add=TRUE)
  fit <- pls_fit(Xtr, ytr, ncomp=3, backend="arma", algorithm="klogitpls")
  pr  <- predict(fit, Xte)
  p1  <- pr[,2]
  auc_like <- cor(p1, yte) # cheap monotone metric
  expect_true(!is.na(auc_like) && auc_like > 0.5)
})
