library(testthat)

skip_if_not_installed("RcppArmadillo")
skip_if_not_installed("pls")

set.seed(123)
n <- 40
p <- 6
m <- 1
X <- matrix(rnorm(n * p), nrow = n)
coef_mat <- matrix(runif(p * m, -1, 1), nrow = p)
Y <- X %*% coef_mat + matrix(rnorm(n * m, sd = 0.1), nrow = n)

fit_simpls <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "simpls", scores = "r", mode = "pls1")
fit_nipals <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "nipals", scores = "r", mode = "pls1")
pls_simpls <- pls::simpls.fit(X,Y,3)
pls_nipals <- pls::oscorespls.fit(X,Y,3)

test_that("pls_fit_nipals matches SIMPLS coefficients", {
  expect_s3_class(fit_nipals, "big_plsr")
  expect_equal(fit_nipals$ncomp, fit_simpls$ncomp)
  expect_equal(fit_simpls$coefficients[,],pls_simpls$coefficients[,,3], tolerance = 1e-6)
  expect_equal(fit_nipals$coefficients[,],pls_nipals$coefficients[,,3], tolerance = 1e-6)
  expect_equal(as.vector(fit_nipals$scores),as.vector(pls_nipals$scores), tolerance = 1e-6)
})

test_that("predict.big_plsr works for responses and scores", {
  preds <- predict(fit_nipals, X, ncomp = 2)
  expect_equal(dim(preds), NULL)
  expect_equal(length(preds), n)
  scores <- predict(fit_nipals, X, ncomp = 2, type = "scores")
  expect_equal(dim(scores), c(n, 2))
})

test_that("summary.big_plsr and VIP return expected structures", {
  summ <- summary(fit_nipals, X, Y)
  expect_s3_class(summ, "summary.big_plsr")
  expect_equal(length(summ$vip), p)
  expect_false(any(is.na(summ$rmse)))
})

test_that("plot helpers run without error", {
  skip_on_ci()
  tmp <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  expect_error(plot_pls_individuals(fit_nipals, comps = c(1, 2)), NA)
  expect_error(plot_pls_variables(fit_nipals, comps = c(1, 2)), NA)
  expect_error(plot_pls_biplot(fit_nipals, comps = c(1, 2)), NA)
  grDevices::dev.off()
})

test_that("plot helpers run without error", {
  skip_on_ci()
  tmp <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  expect_error(plot_pls_individuals(fit_simpls, comps = c(1, 2)), NA)
  expect_error(plot_pls_variables(fit_simpls, comps = c(1, 2)), NA)
  expect_error(plot_pls_biplot(fit_simpls, comps = c(1, 2)), NA)
  grDevices::dev.off()
})

set.seed(123)
n <- 40
p <- 6
m <- 2
X <- matrix(rnorm(n * p), nrow = n)
coef_mat <- matrix(runif(p * m, -1, 1), nrow = p)
Y <- X %*% coef_mat + matrix(rnorm(n * m, sd = 0.1), nrow = n)

fit_simpls <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "simpls", scores = "r", mode = "pls2")
fit_nipals <- pls_fit(X, Y, ncomp = 3, backend = "arma", algorithm = "nipals", scores = "r", mode = "pls2")

expect_equal(fit_simpls$coefficients[,],pls::simpls.fit(X,Y,3)$coefficients[,,3])
pls::oscorespls.fit(X,Y,3)$coefficients[,,3]


test_that("pls_fit_nipals matches SIMPLS coefficients", {
  expect_s3_class(fit_nipals, "big_plsr")
  expect_equal(fit_nipals$ncomp, fit_simpls$ncomp)
  expect_equal(fit_nipals$coefficients, fit_simpls$coefficients, tolerance = 1e-6)
  expect_equal(fit_nipals$intercept,    fit_simpls$intercept,    tolerance = 1e-7)
})

test_that("predict.big_plsr works for responses and scores", {
  preds <- predict(fit_nipals, X, ncomp = 2)
  expect_equal(dim(preds), c(n, m))
  scores <- predict(fit_nipals, X, ncomp = 2, type = "scores")
  expect_equal(dim(scores), c(n, 2))
})

test_that("summary.big_plsr and VIP return expected structures", {
  summ <- summary(fit_nipals, X, Y)
  expect_s3_class(summ, "summary.big_plsr")
  expect_equal(length(summ$vip), p)
  expect_false(any(is.na(summ$rmse)))
})

test_that("plot helpers run without error", {
  skip_on_ci()
  tmp <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  expect_error(plot_pls_individuals(fit_nipals, comps = c(1, 2)), NA)
  expect_error(plot_pls_variables(fit_nipals, comps = c(1, 2)), NA)
  expect_error(plot_pls_biplot(fit_nipals, comps = c(1, 2)), NA)
  grDevices::dev.off()
})

test_that("plot helpers run without error", {
  skip_on_ci()
  tmp <- tempfile(fileext = ".pdf")
  grDevices::pdf(tmp)
  expect_error(plot_pls_individuals(fit_simpls, comps = c(1, 2)), NA)
  expect_error(plot_pls_variables(fit_simpls, comps = c(1, 2)), NA)
  expect_error(plot_pls_biplot(fit_simpls, comps = c(1, 2)), NA)
  grDevices::dev.off()
})

test_that("component selection helpers operate", {
  sel <- pls_select_components(fit_nipals, X, Y, criteria = c("aic", "bic"))
  expect_true(all(c("table", "best") %in% names(sel)))
  expect_equal(ncol(sel$table), 5)
  chosen <- pls_cv_select(pls_cross_validate(X, Y, ncomp = 2, folds = 3, algorithm = "nipals"), metric = "rmse")
  expect_true(chosen %in% 1:2)
})

test_that("bootstrap and threshold utilities work", {
  boot <- pls_bootstrap(X, Y, ncomp = 2, R = 10, algorithm = "nipals", seed = 123)
  expect_equal(dim(boot$mean), c(p, m))
  thr <- pls_threshold(fit_nipals, threshold = 1e-3)
  expect_s3_class(thr, "big_plsr")
})
