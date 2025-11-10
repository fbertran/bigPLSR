# bigPLSR 0.6.8

* Added optional `future`-powered parallel execution to `pls_cross_validate()`
  and `pls_bootstrap()`.
* Extended `pls_bootstrap()` with (X, Y) and (X, T) strategies, percentile and
  BCa confidence intervals, numerical summaries, and coefficient boxplots.
* Added group-aware score plotting with confidence ellipses in
  `plot_pls_individuals()`.
* Added vignettes covering cross-validation/information-criteria workflows and
  bootstrap diagnostics.
  
# bigPLSR 0.6.7
* kernelpls on backend='bigmem' now uses streaming XXᵗ/column paths; the previous 
  dense fallback was removed. Control with options(bigPLSR.kpls_gram = 'rows'|'cols'|'auto') 
  and bigPLSR.chunk_rows, bigPLSR.chunk_cols.

# bigPLSR 0.6.6
* Vignettes: *Kernel and Streaming PLS Methods*, *Automatic Algorithm Selection*.
* Stub C++ entry points for RKHS / kernel logistic / sparse KPLS / KF-PLS.

# bigPLSR 0.6.5

* Algorithm auto-selection: new internal heuristic chooses among
  - **XtX SIMPLS** (standard cross-product SIMPLS),
  - **XXt ("widekernelpls")** for n << p,
  - **NIPALS** when memory is tight or rank is low.
  Tuned by `options(bigPLSR.mem_budget_gb = 8)`. Users can override with `algorithm=`.
* Kernel-style PLS routes: `algorithm = "kernelpls"` and `algorithm = "widekernelpls"`
  implementing Dayal & MacGregor–style (1997) kernel PLS in X-space and wide-X (XXᵗ) space.
* Implemented high-performance kernel and wide-kernel PLS algorithms in
  `pls_fit()` for both dense and bigmemory backends using RcppArmadillo.
* Introduced optional coefficient thresholding.
* Added fast-running examples to all exported functions to improve documentation
  usability on CRAN.

# bigPLSR 0.6.4

* Added kernel PLS and wide-kernel PLS algorithms to `pls_fit()` for both dense
  and bigmemory backends.
* Refreshed plotting helpers with correlation circles, arrow-based loadings and
  a dedicated VIP bar plot.
* Introduced convenience prediction wrappers, information-criteria helpers, and
  expanded cross-validation/bootstrapping utilities to support the new
  algorithms.
* Improved summaries with explained-variance reporting and updated package
  documentation.

# bigPLSR 0.6.2

* Added cross validation and bootstrap for plsR.

# bigPLSR 0.6.1

* Added plots and summaries for `pls_fit()`.

# bigPLSR 0.6.0 

* Added unified path `pls_fit()` for plsR regression that features : dense and bigmemory,
simpls and nipals.

# bigPLSR 0.5.0 

* Added several plsR implementations. Benchmarks.

# bigPLSR 0.4.0 

* Maintainer email update
* Added unit tests

# bigPLSR 0.3.0 

* Code update

# bigPLSR 0.2.0 

* Improving code and help pages

# bigPLSR 0.1.0 

* Implementing gpls, sgpls based models

# bigPLSR 0.0.1 

* Package creation

