# Single-response partial least squares regression (PLS1) another implementation

These helpers wrap high-performance C++ routines built on top of the
`bigmemory` and `bigalgebra` infrastructure. The `pls1_dense_ya`
function performs a standard PLS regression using a NIPALS-style
algorithm without copying the data in memory. The `pls1_stream_ya`
variant iterates over the data in blocks which makes it possible to
handle out-of-core datasets efficiently.

## Usage

``` r
pls1_dense_a(
  X,
  y,
  ncomp = 2L,
  center = TRUE,
  scale = FALSE,
  tol = 1e-08,
  max_iter = 100L,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)

pls1_stream_a(
  X,
  y,
  ncomp = 2L,
  chunk_size = 1024L,
  center = TRUE,
  scale = FALSE,
  tol = 1e-08,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)
```

## Arguments

- X:

  A `big.matrix` object containing the predictors.

- y:

  Either a `big.matrix` with a single column or a numeric vector with
  the response values.

- ncomp:

  Number of latent components to extract.

- center:

  Logical; should the predictors and response be centered.

- scale:

  Logical; should the predictors and response be scaled to unit variance
  before fitting the model.

- tol:

  Numerical tolerance used to detect convergence breakdown.

- max_iter:

  Maximum number of iterations for the internal solver (kept for
  compatibility; the solver adapts automatically when convergence issues
  are detected).

- algorithm:

  Algorithm used to compute the PLS fit. Either "simpls" or "nipals".
  The SIMPLS backend only supports the default centering and scaling
  configuration.

- return_big:

  Logical; when `TRUE`, the coefficients, scores and loadings are
  returned as
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  objects. Defaults to `FALSE`.

- chunk_size:

  Number of rows processed at a time by the streaming backend.

## Value

A list with regression coefficients, intercept, latent scores, weights
and additional metadata.

## Examples

``` r
# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
fit <- pls1_dense_a(X, y, ncomp = 3)
str(fit)
#> List of 16
#>  $ coefficients: num [1:20, 1] -0.0168 -0.1308 0.1621 0.1499 0.0418 ...
#>  $ intercept   : num 0.222
#>  $ x_weights   : num [1:20, 1:3] -0.0027 -0.27846 0.35683 0.41143 0.00574 ...
#>  $ x_loadings  : num [1:20, 1:3] 0.0337 -0.2535 0.3214 0.4333 -0.0755 ...
#>  $ y_loadings  : num [1:3, 1] 0.3218 0.1069 0.0493
#>  $ x_means     : num [1:20] 0.0265 -0.0546 -0.0189 0.0167 -0.1192 ...
#>  $ y_mean      : num 0.192
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] -0.0027 -0.27846 0.35683 0.41143 0.00574 ...
#>  $ loadings    : num [1:20, 1:3] 0.0337 -0.2535 0.3214 0.4333 -0.0755 ...
#>  $ x_center    : num [1:20] 0.0265 -0.0546 -0.0189 0.0167 -0.1192 ...
#>  $ y_center    : num 0.192
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
#>  $ call        : language pls1_dense_a(X = X, y = y, ncomp = 3)
#>  - attr(*, "class")= chr [1:2] "big_plsr" "list"
# }

# \donttest{
library(bigmemory)
X <- as.big.matrix(matrix(rnorm(2000), nrow = 100))
y <- as.big.matrix(matrix(rnorm(100), ncol = 1))
fit <- pls1_stream_a(X, y, ncomp = 3)
str(fit)
#> List of 17
#>  $ coefficients: num [1:20, 1] -0.2457 0.0752 0.1336 -0.2089 0.1301 ...
#>  $ intercept   : num 0.0927
#>  $ x_weights   : num [1:20, 1:3] -0.2551 0.0453 0.2238 -0.2733 0.097 ...
#>  $ x_loadings  : num [1:20, 1:3] -0.1273 -0.0333 0.2768 -0.195 -0.0221 ...
#>  $ y_loadings  : num [1:3, 1] 0.579 0.197 0.1
#>  $ x_means     : num [1:20] -0.0705 -0.0676 0.0993 0.218 -0.1831 ...
#>  $ y_mean      : num -0.0159
#>  $ ncomp       : int 3
#>  $ weights     : num [1:20, 1:3] -0.2551 0.0453 0.2238 -0.2733 0.097 ...
#>  $ loadings    : num [1:20, 1:3] -0.1273 -0.0333 0.2768 -0.195 -0.0221 ...
#>  $ x_center    : num [1:20] -0.0705 -0.0676 0.0993 0.218 -0.1831 ...
#>  $ y_center    : num -0.0159
#>  $ x_scale     : NULL
#>  $ y_scale     : NULL
#>  $ scores      : NULL
#>  $ chunk_size  : int 1024
#>  $ call        : language pls1_stream_a(X = X, y = y, ncomp = 3)
#>  - attr(*, "class")= chr [1:2] "big_plsr" "list"
# }
```
