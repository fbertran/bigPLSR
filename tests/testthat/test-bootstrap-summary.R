skip_on_cran()
skip_on_ci() 

set.seed(123)
X <- matrix(rnorm(80), nrow = 20)
y <- X[, 1] - 0.4 * X[, 2] + rnorm(20, sd = 0.1)

boot_res <- pls_bootstrap(X, y, ncomp = 2, R = 8, type = "xy", parallel = "none")

summ_df <- summarise_pls_bootstrap(boot_res)

test_that("bootstrap summary returns expected columns", {
  expect_s3_class(summ_df, "data.frame")
  expect_true(all(c("mean", "sd", "percentile_lower", "bca_upper") %in% names(summ_df)))
})

test_that("bootstrap object stores BCa intervals", {
  expect_true(!is.null(boot_res$bca_lower))
  expect_equal(dim(boot_res$bca_lower), dim(boot_res$mean))
})

test_that("coefficient boxplot helper runs", {
  expect_silent(plot_pls_bootstrap_coefficients(boot_res))
})