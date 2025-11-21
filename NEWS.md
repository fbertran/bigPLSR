# bigPLSR 0.7.2

* Code and documentation fixes requested by CRAN.

# bigPLSR 0.7.1

* New tuning option: `options(bigPLSR.stream.block_align = 8192L)`. All streamed
  backends (bigmem SIMPLS, streamed scores, RKHS/klogitpls Gram passes, and
  bigmem predict) round their `chunk_size` *up* to a multiple of this alignment,
  then clamp to the available number of rows. Typical sweet spots are 4096–16384
  on modern CPUs.
* If you always need scores on disk, prefer `scores = "big"` to avoid large R
  dense allocations; it streams directly into a `big.matrix`.
* Added benchmarks results and analysis as two vignettes.

# bigPLSR 0.7.0

* Added `plot_pls_bootstrap_scores()` and group-aware ellipses for
  `plot_pls_biplot()` to visualise latent structures.
* Exposed `bigPLSR_stream_kstats()` for streamed RKHS centering statistics and
  corrected the bigmemory RKHS interface to accept dense response blocks.

# bigPLSR 0.6.9

* Stabilised kernel logistic PLS class weighting, reinstated IRLS fallbacks and
  improved dense/big-memory parity.
* Reworked the Kalman-filter state helper to reuse the SIMPLS backend, ensuring
  identical coefficients/intercepts to batch fits.
* Added dedicated RKHS/RKHS-XY and plotting vignettes, and refreshed the PLS1/PLS2
  benchmarking guides with notes on the new algorithms and parallel helpers.
  
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
* Refreshed plotting helpers with variable plots, arrow-based loadings and
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

