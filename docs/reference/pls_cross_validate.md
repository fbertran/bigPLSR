# Cross-validate PLS models

Cross-validate PLS models

## Usage

``` r
pls_cross_validate(
  X,
  Y,
  ncomp,
  folds = 5L,
  type = c("kfold", "loo"),
  algorithm = c("simpls", "nipals", "kernelpls", "widekernelpls"),
  backend = "arma",
  metrics = c("rmse", "mae", "r2"),
  seed = NULL,
  parallel = c("none", "future"),
  future_seed = TRUE,
  ...
)
```

## Arguments

- X:

  Predictor matrix as accepted by
  [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md)

- Y:

  Response matrix or vector as accepted by
  [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md)

- ncomp:

  Integer; components grid to evaluate.

- folds:

  Number of folds (ignored when `type = "loo"`).

- type:

  Either "kfold" (default) or "loo".

- algorithm:

  Backend algorithm: "simpls", "nipals", "kernelpls" or "widekernelpls".

- backend:

  Backend passed to
  [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md).

- metrics:

  Metrics to compute (subset of "rmse", "mae", "r2").

- seed:

  Optional seed for reproducibility.

- parallel:

  Logical or character; same semantics as in
  [`pls_bootstrap()`](https://fbertran.github.io/bigPLSR/reference/pls_bootstrap.md).

- future_seed:

  Logical or integer; reproducible seeds for parallel evaluation.

- ...:

  Passed to
  [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md).

## Value

A list containing per-fold metrics and their summary across folds.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
pls_cross_validate(X, y, ncomp = 2, folds = 3)
#> $details
#>    fold ncomp metric      value
#> 1     1     1   rmse 0.16328299
#> 2     1     1    mae 0.12805868
#> 3     1     1     r2 0.97080073
#> 4     1     2   rmse 0.07303034
#> 5     1     2    mae 0.05597072
#> 6     1     2     r2 0.99415887
#> 7     2     1   rmse 0.24657144
#> 8     2     1    mae 0.20919627
#> 9     2     1     r2 0.95049915
#> 10    2     2   rmse 0.11001802
#> 11    2     2    mae 0.08976574
#> 12    2     2     r2 0.99014504
#> 13    3     1   rmse 0.36542334
#> 14    3     1    mae 0.34056432
#> 15    3     1     r2 0.78937830
#> 16    3     2   rmse 0.38837465
#> 17    3     2    mae 0.37082266
#> 18    3     2     r2 0.76209023
#> 
#> $summary
#>   ncomp metric     value
#> 1     1    mae 0.2259398
#> 2     2    mae 0.1721864
#> 3     1     r2 0.9035594
#> 4     2     r2 0.9154647
#> 5     1   rmse 0.2584259
#> 6     2   rmse 0.1904743
#> 
```
