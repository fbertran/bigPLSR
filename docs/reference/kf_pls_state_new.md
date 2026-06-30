# KF-PLS streaming state (constructor)

Create a persistent Kalman–filter PLS (KF-PLS) state that accumulates
cross-products from streaming mini-batches and later produces a
`big_plsr`-compatible fit via
[`kf_pls_state_fit()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_fit.md).

## Usage

``` r
kf_pls_state_new(p, m, ncomp, lambda = 0.99, q_proc = 0, r_meas = 0)
```

## Arguments

- p:

  Integer, number of predictors (columns of `X`).

- m:

  Integer, number of responses (columns of `Y`).

- ncomp:

  Integer, number of latent components to extract at fit time.

- lambda:

  Numeric in (0,1\], forgetting factor (closer to 1 = slower decay).

- q_proc:

  Non-negative numeric, process-noise magnitude (adds a ridge to
  \\C\_{xx}\\ each update; useful for stabilizing ill-conditioned
  problems).

- r_meas:

  Reserved measurement-noise parameter (not used by the minimal API yet;
  kept for forward compatibility).

## Value

An external pointer to an internal KF-PLS state (opaque object) that you
pass to
[`kf_pls_state_update()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_update.md)
and then to
[`kf_pls_state_fit()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_fit.md)
to produce model coefficients.

## Details

The state maintains exponentially weighted cross-moments \\C\_{xx}\\ and
\\C\_{xy}\\ with forgetting factor `lambda`. When `lambda >= 0.999999`
and `q_proc == 0`, the backend switches to an *exact* accumulation mode
that matches concatenating all chunks (no decay).

## See also

[`kf_pls_state_update()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_update.md),
[`kf_pls_state_fit()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_fit.md),
[`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md)
(use `algorithm = "kf_pls"` for the one-shot dense path).

## Examples

``` r
set.seed(1)
n <- 1000; p <- 50; m <- 2
X1 <- matrix(rnorm(n/2 * p), n/2, p)
X2 <- matrix(rnorm(n/2 * p), n/2, p)
B  <- matrix(rnorm(p*m), p, m)
Y1 <- scale(X1, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n/2*m), n/2, m)
Y2 <- scale(X2, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n/2*m), n/2, m)

st <- kf_pls_state_new(p, m, ncomp = 4, lambda = 0.99, q_proc = 1e-6)
kf_pls_state_update(st, X1, Y1)
kf_pls_state_update(st, X2, Y2)
fit <- kf_pls_state_fit(st)          # returns a big_plsr-compatible list
preds <- predict(bigPLSR::.finalize_pls_fit(fit, "kf_pls"), rbind(X1, X2))
head(preds)
#>             [,1]      [,2]
#> [1,]   4.1992830  3.288988
#> [2,] -10.4765869 11.827720
#> [3,]   3.5877961  6.767181
#> [4,]   0.8577062  4.270630
#> [5,]   4.9413407  4.763492
#> [6,]  -1.7790168 -2.148865
```
