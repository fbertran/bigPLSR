# Partial least squares regression for multi-response problems (PLS2)

Partial least squares regression for multi-response problems (PLS2)

## Usage

``` r
pls2_dense(
  X,
  Y,
  ncomp,
  center = TRUE,
  scale = FALSE,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)

pls2_stream(
  X,
  Y,
  ncomp,
  center = TRUE,
  scale = FALSE,
  chunk_size = 1024L,
  algorithm = c("simpls", "nipals"),
  return_big = FALSE
)
```

## Arguments

- X:

  A
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  containing the predictor variables.

- Y:

  A
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  storing the multi-dimensional response.

- ncomp:

  Number of latent components to compute.

- center:

  Should the inputs be centered prior to fitting?

- scale:

  Should the inputs be scaled to unit variance prior to fitting?

- algorithm:

  PLS backend to use. Either "simpls" (default) or "nipals".

- return_big:

  Logical; when `TRUE`, the coefficients, scores and loadings are
  returned as
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  objects. Defaults to `FALSE`.

- chunk_size:

  Number of rows processed per block by the streaming variant.

## Value

A list with regression coefficients, intercept, weights, loadings and
preprocessing metadata.
