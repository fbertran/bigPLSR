# Streamed centering statistics for RKHS kernels

Compute the column means and grand mean of the kernel matrix \\K(X, X)\\
without materialising it in memory. The input design matrix must be
stored as a
[`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
(or descriptor), and the kernel is evaluated by iterating over
row/column chunks.

## Usage

``` r
bigPLSR_stream_kstats(
  Xbm,
  kernel,
  gamma,
  degree,
  coef0,
  chunk_rows = getOption("bigPLSR.predict.chunk_rows", 8192L),
  chunk_cols = getOption("bigPLSR.predict.chunk_cols", 8192L)
)
```

## Arguments

- Xbm:

  A
  [`bigmemory::big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
  (or descriptor) containing the training design matrix.

- kernel:

  Kernel name passed to
  [`stats::kernel()`](https://rdrr.io/r/stats/kernel.html) compatible
  helpers (`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"`).

- gamma, degree, coef0:

  Kernel hyper-parameters.

- chunk_rows, chunk_cols:

  Numbers of rows/columns to process per chunk.

## Value

A list with entries `r` (column means) and `g` (grand mean) of the
kernel matrix.
