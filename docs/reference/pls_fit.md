# Unified PLS fit with auto backend and selectable algorithm

Dispatches to a dense (Arm/BLAS) backend for in-memory matrices or to a
streaming big.matrix backend when X (or Y) is a big.matrix. Algorithm
can be chosen between: "simpls" (default), "nipals", "kernelpls",
"widekernelpls", "rkhs" (Rosipal & Trejo), "klogitpls", "sparse_kpls",
"rkhs_xy" (double RKHS), and "kf_pls" (Kalman-filter PLS, streaming).

The "kernelpls" paths now include a streaming XX' variant for big.matrix
inputs, with an optional row-chunking loop controlled by `chunk_cols`.

## Usage

``` r
pls_fit(
  X,
  y,
  ncomp,
  tol = 1e-08,
  backend = c("auto", "arma", "bigmem"),
  mode = c("auto", "pls1", "pls2"),
  algorithm = c("auto", "simpls", "nipals", "kernelpls", "widekernelpls", "rkhs",
    "klogitpls", "sparse_kpls", "rkhs_xy", "kf_pls"),
  scores = c("none", "r", "big"),
  chunk_size = 10000L,
  chunk_cols = NULL,
  scores_name = "scores",
  scores_target = c("auto", "new", "existing"),
  scores_bm = NULL,
  scores_backingfile = NULL,
  scores_backingpath = NULL,
  scores_descriptorfile = NULL,
  scores_colnames = NULL,
  return_scores_descriptor = FALSE,
  coef_threshold = NULL,
  kernel = c("linear", "rbf", "poly", "sigmoid"),
  gamma = 1,
  degree = 3L,
  coef0 = 0,
  approx = c("none", "nystrom", "rff"),
  approx_rank = NULL,
  class_weights = NULL
)
```

## Arguments

- X:

  numeric matrix or
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)

- y:

  numeric vector/matrix or `big.matrix`

- ncomp:

  number of latent components

- tol:

  numeric tolerance used in the core solver

- backend:

  one of `"auto"`, `"arma"`, `"bigmem"`

- mode:

  one of `"auto"`, `"pls1"`, `"pls2"`

- algorithm:

  one of `"auto"`, `"simpls"`, `"nipals"`, `"kernelpls"`,
  `"widekernelpls"`, `"rkhs"`, `"klogitpls"`, `"sparse_kpls"`,
  `"rkhs_xy"`, `"kf_pls"`

- scores:

  one of `"none"`, `"r"`, `"big"`

- chunk_size:

  chunk size for the bigmem backend

- chunk_cols:

  columns chunk size for the bigmem backend

- scores_name:

  name for dense scores (or output big.matrix)

- scores_target:

  one of `"auto"`, `"new"`, `"existing"`

- scores_bm:

  optional existing big.matrix or descriptor for scores

- scores_backingfile:

  Character; file name for file-backed scores (when `scores="big"`).

- scores_backingpath:

  Character; directory for the file-backed scores. Defaults to
  [`getwd()`](https://rdrr.io/r/base/getwd.html) or
  [`tempdir()`](https://rdrr.io/r/base/tempfile.html) in streamed
  predict, unless overridden.

- scores_descriptorfile:

  Character; descriptor file name for the file-backed scores.

- scores_colnames:

  optional character vector for score column names

- return_scores_descriptor:

  logical; if TRUE and scores is big.matrix, add `$scores_descriptor`

- coef_threshold:

  Optional non-negative value used to hard-threshold the fitted
  coefficients after model estimation. When supplied, absolute
  coefficients strictly below the threshold are set to zero via
  [`pls_threshold()`](https://fbertran.github.io/bigPLSR/reference/pls_threshold.md).

- kernel:

  kernel name for RKHS/KPLS (`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"`)

- gamma:

  RBF/sigmoid/poly scale parameter

- degree:

  polynomial degree

- coef0:

  polynomial/sigmoid bias

- approx:

  kernel approximation: `"none"`, `"nystrom"`, `"rff"`

- approx_rank:

  rank (columns / features) for the approximation

- class_weights:

  optional numeric weights for classes in `klogitpls`

## Value

a list with coefficients, intercept, weights, loadings, means, and
optionally `$scores`.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r", algorithm = "simpls")
head(pls_predict_response(fit, X, ncomp = 2))
#> [1] -0.2557041 -0.3103345  1.8935717  0.1961492  0.2217772  2.3503614
```
