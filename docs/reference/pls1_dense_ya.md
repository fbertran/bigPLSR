# Single-response partial least squares regression (PLS1) yet another implementation

Single-response partial least squares regression (PLS1) yet another
implementation

## Usage

``` r
pls1_dense_ya(
  x,
  y,
  ncomp,
  tol = 1e-08,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)

pls1_stream_ya(
  x,
  y,
  ncomp,
  chunk_size = 4096,
  tol = 1e-08,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)
```

## Arguments

- x, y:

  Predictor and response objects stored as double precision
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  instances. The response must contain a single column. The dense helper
  also accepts numeric vectors for `y` and converts them transparently.
  The dense routine copies the predictors into an R matrix, while the
  streaming version accesses them in blocks.

- ncomp:

  Number of latent components to extract.

- tol:

  Convergence tolerance used when estimating each component. Only
  relevant for the dense variant.

- algorithm:

  Algorithm used to compute the PLS fit. Either "simpls" or "nipals".
  The SIMPLS backend is generally faster when the data fits in memory.

- return_big:

  Logical; when `TRUE`, the coefficients, scores and loadings are
  returned as
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  objects. Defaults to `FALSE`.

- chunk_size:

  Number of rows processed per block by the streaming variant.

## Value

A list containing regression coefficients, intercept, loadings and
preprocessing statistics. The structure matches the output of the
underlying C++ routines.

## Examples

``` r
# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
fit <- pls1_dense_ya(X, y, ncomp = 3)
str(fit)
#> List of 16
#>  $ coefficients: num [1:20, 1] 0.032576 0.063625 -0.000905 -0.055482 0.097963 ...
#>  $ intercept   : num -0.135
#>  $ x_weights   : num [1:20, 1:3] -0.03604 0.16865 -0.00802 -0.01691 0.16181 ...
#>  $ x_loadings  : num [1:20, 1:3] -0.108728 0.196599 0.000502 0.065538 0.092192 ...
#>  $ y_loadings  : num [1:3, 1] 0.3892 0.119 0.0509
#>  $ x_means     : num [1:20] 4.29e-02 -6.20e-02 5.56e-05 -4.11e-04 1.79e-01 ...
#>  $ y_mean      : num -0.1
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] -0.03604 0.16865 -0.00802 -0.01691 0.16181 ...
#>  $ loadings    : num [1:20, 1:3] -0.108728 0.196599 0.000502 0.065538 0.092192 ...
#>  $ x_center    : num [1:20] 4.29e-02 -6.20e-02 5.56e-05 -4.11e-04 1.79e-01 ...
#>  $ y_center    : num -0.1
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
#>  $ call        : language pls1_dense_ya(x = X, y = y, ncomp = 3)
#>  - attr(*, "class")= chr [1:2] "big_plsr" "list"
# }

# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
fit <- pls1_stream_ya(X, y, ncomp = 3)
str(fit)
#> List of 16
#>  $ coefficients: num [1:20, 1] 0.1189 0.0449 -0.0474 -0.0133 0.0545 ...
#>  $ intercept   : num -0.215
#>  $ x_weights   : num [1:20, 1:3] 0.2465 0.1987 -0.2812 0.1544 0.0542 ...
#>  $ x_loadings  : num [1:20, 1:3] 0.1625 0.2773 -0.3115 0.2517 0.0276 ...
#>  $ y_loadings  : num [1:3, 1] 0.3138 0.1438 0.0843
#>  $ x_means     : num [1:20] 0.0192 0.1354 -0.0974 -0.19 -0.0393 ...
#>  $ y_mean      : num -0.222
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] 0.2465 0.1987 -0.2812 0.1544 0.0542 ...
#>  $ loadings    : num [1:20, 1:3] 0.1625 0.2773 -0.3115 0.2517 0.0276 ...
#>  $ x_center    : num [1:20] 0.0192 0.1354 -0.0974 -0.19 -0.0393 ...
#>  $ y_center    : num -0.222
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
#>  $ call        : language pls1_stream_ya(x = X, y = y, ncomp = 3)
#>  - attr(*, "class")= chr [1:2] "big_plsr" "list"
# }
```
