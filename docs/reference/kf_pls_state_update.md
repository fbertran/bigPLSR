# Update a KF-PLS streaming state with a mini-batch

Feed one chunk (`X_chunk`, `Y_chunk`) to an existing KF-PLS state
created by
[`kf_pls_state_new()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_new.md).
The function updates exponentially weighted means and cross-products (or
exact sufficient statistics when in exact mode).

## Usage

``` r
kf_pls_state_update(state, X_chunk, Y_chunk)
```

## Arguments

- state:

  External pointer produced by
  [`kf_pls_state_new()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_new.md).

- X_chunk:

  Numeric matrix with the same number of columns `p` used to create the
  state.

- Y_chunk:

  Numeric matrix with `m` columns (or a numeric vector if `m == 1`).
  Must have the same number of rows as `X_chunk`.

## Value

Invisibly returns `state`, updated in place.

## Details

Call this repeatedly for each incoming batch. When you want model
coefficients (weights/loadings/intercepts), call
[`kf_pls_state_fit()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_fit.md),
which solves SIMPLS on the accumulated cross-moments without
re-materializing all past data.

## See also

[`kf_pls_state_new()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_new.md),
[`kf_pls_state_fit()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_fit.md)
