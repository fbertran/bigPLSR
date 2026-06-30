# bigPLSR-package

Provides Partial least squares Regression for big data. It allows for
missing data in the explanatory variables. Repeated k-fold
cross-validation of such models using various criteria. Bootstrap
confidence intervals constructions are also available.

## References

Maumy, M., Bertrand, F. (2023). PLS models and their extension for big
data. Joint Statistical Meetings (JSM 2023), Toronto, ON, Canada.

Maumy, M., Bertrand, F. (2023). bigPLS: Fitting and cross-validating
PLS-based Cox models to censored big data. BioC2023 — The Bioconductor
Annual Conference, Dana-Farber Cancer Institute, Boston, MA, USA.
Poster. https://doi.org/10.7490/f1000research.1119546.1

## See also

Useful links:

- <https://fbertran.github.io/bigPLSR/>

- <https://github.com/fbertran/bigPLSR>

- Report bugs at <https://github.com/fbertran/bigPLSR/issues>

## Author

**Maintainer**: Frederic Bertrand <frederic.bertrand@lecnam.net>
([ORCID](https://orcid.org/0000-0002-0837-8281))

Authors:

- Frederic Bertrand <frederic.bertrand@lecnam.net>
  ([ORCID](https://orcid.org/0000-0002-0837-8281))

- Myriam Maumy <myriam.maumy@ehesp.fr>
  ([ORCID](https://orcid.org/0000-0002-4615-1512))

## Examples

``` r
set.seed(123)
X <- matrix(rnorm(60), nrow = 20)
y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
fit <- pls_fit(X, y, ncomp = 2, scores = "r", algorithm = "simpls")
head(pls_predict_response(fit, X, ncomp = 2))
#> [1] -0.2557041 -0.3103345  1.8935717  0.1961492  0.2217772  2.3503614
```
