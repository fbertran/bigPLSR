# Variable importance in projection (VIP) scores

Variable importance in projection (VIP) scores

## Usage

``` r
pls_vip(object, comps = NULL)
```

## Arguments

- object:

  A fitted PLS model.

- comps:

  Components used to compute the VIP scores. Defaults to all available
  components.

## Value

A named numeric vector of VIP scores.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
pls_vip(fit)
#> [1] 0.3679251 0.2478539 0.1299815 0.1706395
```
