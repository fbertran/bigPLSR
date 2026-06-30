# Select components from cross-validation results

Select components from cross-validation results

## Usage

``` r
pls_cv_select(cv_result, metric = c("rmse", "mae", "r2"), minimise = NULL)
```

## Arguments

- cv_result:

  Result returned by
  [`pls_cross_validate()`](https://fbertran.github.io/bigPLSR/reference/pls_cross_validate.md).

- metric:

  Metric to optimise.

- minimise:

  Logical; whether the metric should be minimised.

## Value

Selected number of components.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
cv <- pls_cross_validate(X, y, ncomp = 2, folds = 3)
pls_cv_select(cv, metric = "rmse")
#> [1] 2
```
