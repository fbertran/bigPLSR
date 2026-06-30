# Predict responses from a PLS fit

Predict responses from a PLS fit

## Usage

``` r
pls_predict_response(object, newdata, ncomp = NULL)
```

## Arguments

- object:

  A fitted PLS model.

- newdata:

  Predictor matrix for scoring.

- ncomp:

  Number of components to use.

## Value

A numeric matrix or vector of predictions.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
pls_predict_response(fit, X, ncomp = 2)
#>  [1] -1.0141790 -0.5820463  1.1675424  0.1269469  0.4234421  0.5985047
#>  [7]  0.3557169 -0.3717287 -1.1921817 -0.2849629
```
