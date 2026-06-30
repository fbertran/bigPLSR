# Predict latent scores from a PLS fit

Predict latent scores from a PLS fit

## Usage

``` r
pls_predict_scores(object, newdata, ncomp = NULL)
```

## Arguments

- object:

  A fitted PLS model.

- newdata:

  Predictor matrix for scoring.

- ncomp:

  Number of components to use.

## Value

Matrix of component scores.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
pls_predict_scores(fit, X, ncomp = 2)
#>                t1          t2
#>  [1,] -0.13544977 -0.52708945
#>  [2,] -0.17098259 -0.14902942
#>  [3,]  0.50972517  0.24632134
#>  [4,]  0.08135113  0.04355323
#>  [5,]  0.08487733  0.26452562
#>  [6,]  0.55636410 -0.25130464
#>  [7,]  0.08732598  0.20957085
#>  [8,] -0.44910850  0.39409855
#>  [9,] -0.26198780 -0.48844206
#> [10,] -0.30211506  0.25779599
```
