# Fast IRLS for binomial logit with class weights

Fast IRLS for binomial logit with class weights

## Usage

``` r
cpp_irls_binomial(TT, ybin, w_class = NULL, maxit = 50L, tol = 1e-08)
```

## Arguments

- TT:

  n x A numeric matrix of latent scores (no intercept column)

- ybin:

  integer vector of {0,1} labels (length n)

- w_class:

  optional length-2 numeric vector: weights for classes c( w0, w1 )

- maxit:

  max IRLS iterations

- tol:

  relative tolerance on parameter change

## Value

list(beta = A-vector, b = scalar intercept, fitted = n-vector, iter =
integer, converged = logical)
