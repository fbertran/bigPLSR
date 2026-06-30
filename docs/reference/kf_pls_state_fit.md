# Finalize a KF-PLS state into a fitted model

Converts the accumulated KF-PLS state into a SIMPLS-equivalent fitted
model (using the current sufficient statistics). The result is
compatible with
[`predict.big_plsr()`](https://fbertran.github.io/bigPLSR/reference/predict.big_plsr.md).

## Usage

``` r
kf_pls_state_fit(state, tol = 1e-08)
```

## Arguments

- state:

  External pointer created by
  [`kf_pls_state_new()`](https://fbertran.github.io/bigPLSR/reference/kf_pls_state_new.md).

- tol:

  Numeric tolerance for the inner SIMPLS step.

## Value

A list with PLS factors and coefficients, classed as `big_plsr`.

## Examples

``` r
n <- 200; p <- 30; m <- 2; A <- 3
X <- matrix(rnorm(n*p), n, p)
Y <- X[,1:2] %*% matrix(c(0.7, -0.3, 0.2, 0.9), 2, m) + matrix(rnorm(n*m, sd=0.2), n, m)

state <- kf_pls_state_new(p, m, A, lambda = 0.99, q_proc = 1e-6)

# stream in mini-batches
bs <- 64
for (i in seq(1, n, by = bs)) {
  idx <- i:min(i+bs-1, n)
  kf_pls_state_update(state, X[idx, , drop=FALSE], Y[idx, , drop=FALSE])
}

fit <- kf_pls_state_fit(state)  # returns a big_plsr-compatible list
# predict via your existing predict.big_plsr (linear case)
Yhat <- cbind(1, scale(X, center = fit$x_means, scale = FALSE)) %*%
  rbind(fit$intercept, fit$coefficients)
```
