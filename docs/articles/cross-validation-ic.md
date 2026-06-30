# Cross-validation and Information Criteria in bigPLSR

## Overview

This vignette illustrates how to evaluate partial least squares (PLS)
models with repeated cross-validation and information criteria using the
new parallel helpers available in `bigPLSR`.

We generate a small synthetic data set so the examples run quickly even
when the vignette is built during package installation.

``` r

library(bigPLSR)
n <- 120; p <- 8
X <- matrix(rnorm(n * p), n, p)
eta <- X[, 1] - 0.8 * X[, 2] + 0.5 * X[, 3]
y <- eta + rnorm(n, sd = 0.4)
```

## Cross-validation

The
[`pls_cross_validate()`](https://fbertran.github.io/bigPLSR/reference/pls_cross_validate.md)
function now accepts a `parallel` argument. Setting
`parallel = "future"` evaluates the folds concurrently by relying on the
[`future`](https://future.futureverse.org/) ecosystem. You are free to
configure any execution plan you like before calling the helper. Below
we keep the sequential default to avoid introducing run-time
dependencies during the build process.

``` r

cv_res <- pls_cross_validate(X, y, ncomp = 4, folds = 6,
                             metrics = c("rmse", "r2"),
                             parallel = "none")
head(cv_res$details)
#>   fold ncomp metric     value
#> 1    1     1   rmse 0.4673779
#> 2    1     1     r2 0.8877468
#> 3    1     2   rmse 0.4176394
#> 4    1     2     r2 0.9103676
#> 5    1     3   rmse 0.3397565
#> 6    1     3     r2 0.9406804
```

Aggregating the metrics provides a quick overview of the predictive
performance per number of components:

``` r

cv_res$summary
#>   ncomp metric     value
#> 1     1     r2 0.8263996
#> 2     2     r2 0.8928828
#> 3     3     r2 0.9039359
#> 4     4     r2 0.9039186
#> 5     1   rmse 0.5430639
#> 6     2   rmse 0.4294906
#> 7     3   rmse 0.4038882
#> 8     4   rmse 0.4038991
```

The cross-validation table is convenient for downstream selection. For
example, we can pick the component count that minimises the RMSE:

``` r

pls_cv_select(cv_res, metric = "rmse")
#> [1] 3
```

## Information criteria

Information criteria complement cross-validation by trading off goodness
of fit with model complexity. The helper
[`pls_information_criteria()`](https://fbertran.github.io/bigPLSR/reference/pls_information_criteria.md)
computes the RSS, RMSE, AIC and BIC across components:

``` r

fit <- pls_fit(X, y, ncomp = 4, scores = "r")
ic_tbl <- pls_information_criteria(fit, X, y)
ic_tbl
#>   ncomp      rss      rmse       aic       bic
#> 1     1 28.81873 0.4900572 -167.1760 -161.6010
#> 2     2 18.63632 0.3940846 -217.4856 -209.1231
#> 3     3 17.49284 0.3818032 -223.0840 -211.9340
#> 4     4 17.39255 0.3807071 -221.7740 -207.8365
```

For convenience the wrapper
[`pls_select_components()`](https://fbertran.github.io/bigPLSR/reference/pls_select_components.md)
selects the best components according to the requested criteria:

``` r

pls_select_components(fit, X, y, criteria = c("aic", "bic"))
#> $table
#>   ncomp      rss      rmse       aic       bic
#> 1     1 28.81873 0.4900572 -167.1760 -161.6010
#> 2     2 18.63632 0.3940846 -217.4856 -209.1231
#> 3     3 17.49284 0.3818032 -223.0840 -211.9340
#> 4     4 17.39255 0.3807071 -221.7740 -207.8365
#> 
#> $best
#> $best$aic
#> [1] 3
#> 
#> $best$bic
#> [1] 3
```

## Parallel execution with `future`

If you wish to parallelise cross-validation, configure a plan before
calling the helper. The example below assumes a multicore environment
and therefore is not run during vignette building:

``` r

future::plan(future::multisession, workers = 2)
cv_parallel <- pls_cross_validate(X, y, ncomp = 4, folds = 6,
                                  metrics = c("rmse", "mae"),
                                  parallel = "future",
                                  future_seed = TRUE)
future::plan(future::sequential)
```

The `future_seed` argument ensures reproducible bootstrap samples even
when multiple workers are used.

## Summary

The refreshed cross-validation workflow exposes a consistent interface
for sequential and parallel execution, while the information-criteria
helpers offer another perspective on component selection. The
combination lets you systematically tune your PLS models for both
accuracy and parsimony.
