# Title

Title

## Usage

``` r
bigscale(
  formula = Surv(time = time, status = status) ~ .,
  data,
  norm.method = "standardize",
  strata.size = 20,
  batch.size = 1,
  features.mean = NULL,
  features.sd = NULL,
  parallel.flag = FALSE,
  num.cores = NULL,
  bigmemory.flag = FALSE,
  num.rows.chunk = 1e+06,
  col.names = NULL,
  type = "short"
)
```

## Arguments

- formula:
- data:
- norm.method:
- strata.size:
- batch.size:
- features.mean:
- features.sd:
- parallel.flag:
- num.cores:
- bigmemory.flag:
- num.rows.chunk:
- col.names:
- type:

## Value

an object of the scaler class time.indices: indices of the time variable
cens.indices: indices of the censored variables features.indices:
indices of the features time.sd: standard deviation of the time variable
time.mean: mean of the time variable features.sd: standard deviation of
the features features.mean: mean of the features nr: number of rows nc:
number of columns col.names: columns names

## Examples

``` r

1+1
#> [1] 2
```
