# Internal: resolve training reference for RKHS predictions

Accepts:

- dense matrix (returned as-is)

- big.matrix (returned as-is)

- big.matrix.descriptor (attached and returned)

## Usage

``` r
.resolve_training_ref(obj, dots)
```

## Details

Sources (priority): object\$X, object\$Xtrain, ...\$Xtrain, ...\$X_ref,
object\$X_ref
