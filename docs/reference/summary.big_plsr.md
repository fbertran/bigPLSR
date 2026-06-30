# Summarize a `big_plsr` model

Summarize a `big_plsr` model

## Usage

``` r
# S3 method for class 'big_plsr'
summary(object, ..., X = NULL, Y = NULL)
```

## Arguments

- object:

  A fitted PLS model.

- ...:

  Unused.

- X:

  Optional design matrix to recompute reconstruction metrics.

- Y:

  Optional response matrix/vector.

## Value

An object of class `summary.big_plsr`.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
summary(fit)
#> Partial least squares regression summary
#> Algorithm: simpls 
#> Mode: pls1 
#> Components: 2 
#> Score variance: 0.1111, 0.1111 
#> Explained variance (%): 50.0, 50.0 
#> VIP (first 10):
#> [1] 0.3679251 0.2478539 0.1299815 0.1706395
```
