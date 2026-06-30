# Print a `summary.big_plsr` object

Print a `summary.big_plsr` object

## Usage

``` r
# S3 method for class 'summary.big_plsr'
print(x, ...)
```

## Arguments

- x:

  A `summary.big_plsr` object.

- ...:

  Passed to lower-level print methods.

## Value

`x`, invisibly.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
print(summary(fit))
#> Partial least squares regression summary
#> Algorithm: simpls 
#> Mode: pls1 
#> Components: 2 
#> Score variance: 0.1111, 0.1111 
#> Explained variance (%): 50.0, 50.0 
#> VIP (first 10):
#> [1] 0.3679251 0.2478539 0.1299815 0.1706395
```
