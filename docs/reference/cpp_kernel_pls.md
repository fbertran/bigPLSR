# Internal kernel and wide-kernel PLS solver

Internal kernel and wide-kernel PLS solver

## Usage

``` r
cpp_kernel_pls(X, Y, ncomp, tol, wide)
```

## Arguments

- X:

  Centered design matrix.

- Y:

  Centered response matrix.

- ncomp:

  Maximum number of components.

- tol:

  Numerical tolerance.

- wide:

  Whether to use the wide-kernel update.

## Value

A list containing the kernel PLS factors.
