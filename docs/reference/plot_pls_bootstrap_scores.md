# Boxplots of bootstrap score distributions

Visualise the variability of latent scores obtained through
[`pls_bootstrap()`](https://fbertran.github.io/bigPLSR/reference/pls_bootstrap.md)
when `return_scores = TRUE`.

## Usage

``` r
plot_pls_bootstrap_scores(
  boot_result,
  components = NULL,
  observations = NULL,
  ...
)
```

## Arguments

- boot_result:

  Result returned by
  [`pls_bootstrap()`](https://fbertran.github.io/bigPLSR/reference/pls_bootstrap.md).

- components:

  Optional vector of component indices or names to include.

- observations:

  Optional vector of observation indices or names to include.

- ...:

  Additional arguments passed to
  [`graphics::boxplot()`](https://rdrr.io/r/graphics/boxplot.html).

## Value

Invisibly returns `NULL` after drawing the boxplots.
