# Component selection via information criteria

Component selection via information criteria

## Usage

``` r
pls_select_components(
  object,
  X,
  Y,
  criteria = c("aic", "bic"),
  max_comp = NULL
)
```

## Arguments

- object:

  A fitted PLS model.

- X:

  Training design matrix.

- Y:

  Training response matrix or vector.

- criteria:

  Character vector specifying which criteria to compute.

- max_comp:

  Maximum number of components to consider.

## Value

A list with the per-component table and the selected components.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
pls_select_components(fit, X, y)
#> $table
#>   ncomp       rss      rmse       aic       bic
#> 1     1 1.1527765 0.2400809 -53.07118 -51.07971
#> 2     2 0.7192385 0.1896363 -60.50589 -57.51869
#> 
#> $best
#> $best$aic
#> [1] 2
#> 
#> $best$bic
#> [1] 2
#> 
#> 
```
