# Matrix and arithmetic operations for big.matrix objects

These methods extend the base matrix multiplication operator
([`%*%`](https://rdrr.io/r/base/matmult.html)) and the group generic
[`Arithmetic`](https://rdrr.io/r/base/Arithmetic.html) so that
[`big.matrix`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html)
objects can interoperate with base R matrices and numeric scalars using
the high-performance routines provided by bigalgebra.

## Usage

``` r
# S4 method for class 'big.matrix,big.matrix'
x %*% y

# S4 method for class 'matrix,big.matrix'
x %*% y

# S4 method for class 'big.matrix,matrix'
x %*% y

# S4 method for class 'big.matrix,big.matrix'
Arith(e1, e2)

# S4 method for class 'big.matrix,matrix'
Arith(e1, e2)

# S4 method for class 'matrix,big.matrix'
Arith(e1, e2)

# S4 method for class 'numeric,big.matrix'
Arith(e1, e2)

# S4 method for class 'big.matrix,numeric'
Arith(e1, e2)
```

## Arguments

- x, y:

  Matrix operands supplied either as `big.matrix` instances or base R
  matrices, depending on the method signature.

- e1, e2:

  Numeric operands, which may be `big.matrix` objects, base R matrices,
  or numeric scalars depending on the method signature.

## Details

Matrix multiplications dispatch to
[`bigalgebra::dgemm()`](https://fbertran.github.io/bigalgebra/reference/dgemm.html),
mixed arithmetic on matrices relies on
[`bigalgebra::daxpy()`](https://fbertran.github.io/bigalgebra/reference/daxpy.html),
and scalar/matrix combinations use
[`bigalgebra::dadd()`](https://fbertran.github.io/bigalgebra/reference/dadd.html)
when appropriate.

## See also

[`bigmemory::big.matrix()`](https://rdrr.io/pkg/bigmemory/man/big.matrix.html),
[`bigalgebra::dgemm()`](https://fbertran.github.io/bigalgebra/reference/dgemm.html),
[`bigalgebra::daxpy()`](https://fbertran.github.io/bigalgebra/reference/daxpy.html),
[`bigalgebra::dadd()`](https://fbertran.github.io/bigalgebra/reference/dadd.html)

## Examples

``` r
if (requireNamespace("bigmemory", quietly = TRUE) &&
    requireNamespace("bigalgebra", quietly = TRUE)) {
  x <- bigmemory::big.matrix(2, 2, init = 1)
  y <- bigmemory::big.matrix(2, 2, init = 2)
  x %*% y
  x + y
  x * 3
}
#> An object of class "big.matrix"
#> Slot "address":
#> <pointer: 0x104485300>
#> 
```
