# Boxplots of bootstrap coefficient distributions

Boxplots of bootstrap coefficient distributions

## Usage

``` r
plot_pls_bootstrap_coefficients(
  boot_result,
  responses = NULL,
  variables = NULL,
  ...
)
```

## Arguments

- boot_result:

  Result returned by
  [`pls_bootstrap()`](https://fbertran.github.io/bigPLSR/reference/pls_bootstrap.md).

- responses:

  Optional character vector selecting response columns.

- variables:

  Optional character vector selecting predictor variables.

- ...:

  Additional arguments passed to
  [`graphics::boxplot()`](https://rdrr.io/r/graphics/boxplot.html).

## Value

Invisibly returns `NULL` after drawing the boxplots.
