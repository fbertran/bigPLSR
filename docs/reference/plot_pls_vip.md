# Plot Variable Importance in Projection (VIP)

Plot Variable Importance in Projection (VIP)

## Usage

``` r
plot_pls_vip(
  object,
  comps = NULL,
  threshold = 1,
  palette = c("#4575b4", "#d73027"),
  ...
)
```

## Arguments

- object:

  A fitted PLS model.

- comps:

  Components to aggregate. Defaults to all available.

- threshold:

  Optional threshold to highlight influential variables.

- palette:

  Colour palette used for bars.

- ...:

  Additional parameters passed to
  [`graphics::barplot()`](https://rdrr.io/r/graphics/barplot.html).

## Value

Invisibly returns the VIP scores used to create the bar plot.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(40), nrow = 10)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(10, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
plot_pls_vip(fit)
```
