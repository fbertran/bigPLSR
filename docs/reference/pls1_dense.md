# Single-response partial least squares regression (PLS1)

These helpers expose optimised dense and streaming solvers tailored for
partial least squares regression problems where the response consists of
a single column. They wrap the high performance C++ routines shipped
with the package and provide a user friendly entry point when
benchmarking the available implementations.

## Usage

``` r
.harmonize_pls_result(res)

pls1_stream(
  X,
  y,
  ncomp = 2L,
  chunk_size = 1024L,
  center = TRUE,
  scale = FALSE,
  center_y = TRUE,
  scale_y = FALSE,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)
```

## Arguments

- X:

  A
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  storing the design matrix.

- y:

  Numeric vector of responses with length `nrow(X)`.

- ncomp:

  Number of latent components to compute.

- chunk_size:

  Number of rows to process per chunk. Must be strictly positive.
  Smaller chunks reduce peak memory usage while larger chunks may
  improve speed.

- center:

  Should the columns of `X` be centered? Defaults to `TRUE`.

- scale:

  Should the columns of `X` be scaled to unit variance? Defaults to
  `FALSE`.

- center_y:

  Should the response be centered? Defaults to `TRUE`.

- scale_y:

  Should the response be scaled to unit variance? Defaults to `FALSE`.

- algorithm:

  Algorithm to use for the fit. Either "simpls" or "nipals". When
  choosing "simpls", preprocessing options must remain at their default
  values.

- return_big:

  Logical; when `TRUE`, the coefficients, scores and loadings are
  returned as
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  objects. Defaults to `FALSE`.

## Value

A list containing regression coefficients, intercept, latent scores,
loadings and weights.

## Examples

``` r
# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- matrix(rnorm(100), ncol = 1)
fit <- pls1_dense(X, y, ncomp = 3)
str(fit)
#> List of 15
#>  $ coefficients: num [1:20, 1] 0.1145 0.1132 -0.0814 -0.1989 0.1312 ...
#>  $ intercept   : num -0.167
#>  $ x_weights   : num [1:20, 1:3] 0.0944 0.093 -0.2491 -0.1897 0.1397 ...
#>  $ x_loadings  : num [1:20, 1:3] 0.03071 0.09135 -0.34936 -0.00304 0.07533 ...
#>  $ y_loadings  : num [1:3, 1] 0.439 0.224 0.1
#>  $ x_means     : num [1:20] -0.0032 -0.07841 -0.16905 -0.08206 -0.00219 ...
#>  $ y_mean      : num -0.091
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] 0.0944 0.093 -0.2491 -0.1897 0.1397 ...
#>  $ loadings    : num [1:20, 1:3] 0.03071 0.09135 -0.34936 -0.00304 0.07533 ...
#>  $ x_center    : num [1:20] -0.0032 -0.07841 -0.16905 -0.08206 -0.00219 ...
#>  $ y_center    : num -0.091
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
# }

# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- matrix(rnorm(100), ncol = 1)
fit <- pls1_stream(X, y, ncomp = 3)
str(fit)
#> List of 15
#>  $ coefficients: num [1:20, 1] -0.0634 -0.0947 0.0649 0.1416 -0.0312 ...
#>  $ intercept   : num -0.0348
#>  $ x_weights   : num [1:20, 1:3] -0.0193 -0.2892 0.0764 0.0932 -0.0772 ...
#>  $ x_loadings  : num [1:20, 1:3] 0.069 -0.3726 0.0123 -0.0112 -0.0616 ...
#>  $ y_loadings  : num [1:3, 1] 0.405 0.194 0.074
#>  $ x_means     : num [1:20] -0.0804 -0.0878 -0.1322 0.104 0.1434 ...
#>  $ y_mean      : num -0.0538
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] -0.0193 -0.2892 0.0764 0.0932 -0.0772 ...
#>  $ loadings    : num [1:20, 1:3] 0.069 -0.3726 0.0123 -0.0112 -0.0616 ...
#>  $ x_center    : num [1:20] -0.0804 -0.0878 -0.1322 0.104 0.1434 ...
#>  $ y_center    : num -0.0538
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
# }
```
