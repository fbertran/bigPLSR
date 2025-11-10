skip_on_cran()

set.seed(42)
X <- matrix(rnorm(40), nrow = 20)
y <- rnorm(20)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")

fac <- gl(2, 10)

test_that("plot_pls_individuals handles grouping ellipses", {
  expect_silent(plot_pls_individuals(fit, groups = fac, ellipse = TRUE))
})