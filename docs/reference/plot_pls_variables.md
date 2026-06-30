# Plot variable loadings

Plot variable loadings

## Usage

``` r
plot_pls_variables(
  object,
  comps = c(1L, 2L),
  circle = TRUE,
  circle_col = "grey80",
  arrow_col = "steelblue",
  arrow_scale = 1,
  ...
)
```

## Arguments

- object:

  A fitted PLS model.

- comps:

  Components to display (length two).

- circle:

  Logical; draw the unit circle.

- circle_col:

  Colour of the unit circle.

- arrow_col:

  Colour of the variable arrows.

- arrow_scale:

  Scaling applied to variable vectors.

- ...:

  Additional plotting parameters passed to
  [`graphics::plot()`](https://rdrr.io/r/graphics/plot.default.html).

## Value

Invisibly returns `NULL` after drawing the plot.

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r")
plot_pls_variables(fit)
```
