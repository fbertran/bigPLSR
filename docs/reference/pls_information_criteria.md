# Compute information criteria for component selection

Compute information criteria for component selection

## Usage

``` r
pls_information_criteria(object, X, Y, max_comp = NULL)
```

## Arguments

- object:

  A fitted PLS model.

- X:

  Training design matrix.

- Y:

  Training response matrix or vector.

- max_comp:

  Maximum number of components to consider.

## Value

A data frame with RSS, RMSE, AIC and BIC per component.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
pls_information_criteria(fit, X, y)
#>   ncomp       rss      rmse       aic       bic
#> 1     1 1.1527765 0.2400809 -53.07118 -51.07971
#> 2     2 0.7192385 0.1896363 -60.50589 -57.51869
```
