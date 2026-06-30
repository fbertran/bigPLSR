# RKHS-based Algorithms in bigPLSR

## Overview

bigPLSR implements two kernel-based partial least squares solvers:

- `algorithm = "rkhs"` (Rosipal & Trejo style) projects only the
  predictor matrix $`X`$ into an RKHS;
- `algorithm = "rkhs_xy"` projects both $`X`$ and the response matrix
  $`Y`$ into (possibly different) RKHSs and couples the latent scores
  through a regularised cross-covariance operator.

Both solvers are available for dense matrices and for
[`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
objects. The big-memory paths stream kernel blocks and persist centering
statistics so predictions remain cheap.

## Dense example

``` r

library(bigPLSR)
set.seed(42)
n <- 120; p <- 8; m <- 2
X <- matrix(rnorm(n * p), n, p)
Y <- cbind(
  sin(X[, 1]) + 0.3 * X[, 2]^2 + rnorm(n, sd = 0.1),
  cos(X[, 3]) - 0.2 * X[, 4] + rnorm(n, sd = 0.1)
)

fit_rkhs <- pls_fit(X, Y, ncomp = 3, algorithm = "rkhs",
                    kernel = "rbf", gamma = 1 / p, scores = "r")

options(bigPLSR.rkhs_xy.lambda_x = 1e-6)
options(bigPLSR.rkhs_xy.lambda_y = 1e-6)

fit_rkhs_xy <- pls_fit(X, Y, ncomp = 3, algorithm = "rkhs_xy",
                       kernel = "rbf", gamma = 1 / p,
                       scores = "none")

head(predict(fit_rkhs, X))
#>              [,1]       [,2]
#> [1,]  1.450177968 0.89078409
#> [2,] -0.007192657 0.54629507
#> [3,]  0.339494159 0.56225998
#> [4,]  0.437162419 0.48400010
#> [5,]  0.427444560 0.42801983
#> [6,] -0.058173062 0.06114628
head(predict(fit_rkhs_xy, X))
#>           [,1]       [,2]
#> [1,] 1.7433526  0.9580851
#> [2,] 0.1298308  0.7311463
#> [3,] 0.2710529  0.3426840
#> [4,] 0.8168251  0.2991740
#> [5,] 0.4341051  0.2350586
#> [6,] 0.1174506 -0.4975300
```

Both fits run in well under five seconds for this moderately sized
example. The RKHS-XY variant stores kernel centering statistics for both
sides so that [`predict()`](https://rdrr.io/r/stats/predict.html) can
re-use them without recomputing the entire Gram matrix.

## Streaming example

``` r

library(bigmemory)
Xbm <- as.big.matrix(X)
Ybm <- as.big.matrix(Y)

fit_stream <- pls_fit(Xbm, Ybm, ncomp = 3, backend = "bigmem",
                      algorithm = "rkhs", kernel = "rbf",
                      gamma = 1 / p, chunk_size = 1024L,
                      scores = "none")
```

The streaming call attaches training descriptors (`$X_ref`) and kernel
centering summaries (`$kstats`) automatically. When
[`predict()`](https://rdrr.io/r/stats/predict.html) is invoked on new
data with `Xtrain = fit_stream$X_ref`, the package streams the
cross-kernel blocks and avoids materialising the full $`n_\text{new}
\times n_\text{train}`$ Gram matrix.

## Logistic response

Kernel logistic PLS (`algorithm = "klogitpls"`) builds on the RKHS
infrastructure. After extracting latent scores from the centered Gram
matrix the algorithm runs a logistic IRLS procedure in score space with
support for class weighting and optional alternating score updates.
Small datasets (hundreds of observations) remain well within the
five-second budget.

``` r

y <- as.integer(X[, 1]^2 + X[, 2]^2 + rnorm(n, sd = 0.2) > 1)
fit_logit <- pls_fit(X, y, ncomp = 2, algorithm = "klogitpls",
                     kernel = "rbf", gamma = 1 / p)
mean(predict(fit_logit, X))
```
