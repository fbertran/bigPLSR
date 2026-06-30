# Predict method for big_plsr objects

Predict method for big_plsr objects

## Usage

``` r
# S3 method for class 'big_plsr'
predict(
  object,
  newdata,
  ncomp = NULL,
  type = c("response", "scores", "prob", "class"),
  ...
)
```

## Arguments

- object:

  A fitted PLS model produced by
  [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md).

- newdata:

  Matrix or
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  with predictor values.

- ncomp:

  Number of components to use for prediction.

- type:

  Either "response" (default) or "scores".

- ...:

  Unused, for compatibility with the generic.

## Value

Predicted responses or component scores.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
predict(fit, X, ncomp = 2)
#>  [1] -1.0141790 -0.5820463  1.1675424  0.1269469  0.4234421  0.5985047
#>  [7]  0.3557169 -0.3717287 -1.1921817 -0.2849629
```
