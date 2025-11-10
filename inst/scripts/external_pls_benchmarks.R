#' Generate external PLS benchmark results
#'
#' This script benchmarks bigPLSR against reference implementations from
#' the pls and mixOmics packages. It is not run during package checks and is
#' provided so that contributors can refresh the pre-computed dataset shipped
#' in `data/external_pls_benchmarks.rda`.
#'
#' Usage:
#'   Rscript inst/scripts/external_pls_benchmarks.R
#'
#' The script saves a data frame called `external_pls_benchmarks` to
#' `data/external_pls_benchmarks.rda`.

suppressPackageStartupMessages({
  library(bigPLSR)
  library(bigmemory)
  library(bench)
  library(pls)
  library(mixOmics)
})

set.seed(42)

make_task <- function(n, p, q = 1) {
  X <- matrix(rnorm(n * p), nrow = n)
  loadings <- matrix(rnorm(p * q), nrow = p)
  scores <- matrix(rnorm(n * q), nrow = n)
  Y <- scale(scores %*% t(loadings[seq_len(q), , drop = FALSE]) +
               matrix(rnorm(n * q, sd = 0.5), nrow = n))
  list(X = X, Y = Y)
}

run_benchmark <- function(task_name, algorithm, n, p, q) {
  dat <- make_task(n, p, q)
  Xbm <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
  Xbm[,] <- dat$X
  Ybm <- bigmemory::big.matrix(nrow = n, ncol = q, type = "double")
  Ybm[,] <- dat$Y
  
  res <- bench::mark(
    bigPLSR = pls_fit(dat$X, dat$Y, ncomp = min(5, q), algorithm = algorithm),
    pls = {
      if (q == 1) {
        pls::plsr(dat$Y ~ dat$X, ncomp = min(5, ncol(dat$X)), method = algorithm)
      } else {
        pls::plsr(dat$Y ~ dat$X, ncomp = min(5, ncol(dat$X)), method = algorithm)
      }
    },
    mixOmics = {
      if (q == 1) {
        mixOmics::pls(dat$X, dat$Y[,1, drop = FALSE], ncomp = min(5, ncol(dat$X)))
      } else {
        mixOmics::spls(dat$X, dat$Y, ncomp = min(5, ncol(dat$X)))
      }
    },
    iterations = 20,
    check = FALSE
  )
  
  data.frame(
    task = task_name,
    algorithm = algorithm,
    package = res$expression,
    median_time_ms = as.numeric(res$median / 1e6),
    itr_per_sec = res$`itr/sec`,
    n = n,
    p = p,
    q = q,
    stringsAsFactors = FALSE
  )
}

benchmarks <- rbind(
  run_benchmark("pls1", "simpls", n = 1000, p = 50, q = 1),
  run_benchmark("pls1", "widekernelpls", n = 1000, p = 50, q = 1),
  run_benchmark("pls2", "simpls", n = 800, p = 60, q = 3)
)

external_pls_benchmarks <- within(benchmarks, {
  notes <- ifelse(package == "bigPLSR",
                  "Run via pls_fit() with dense backend",
                  ifelse(package == "pls",
                         "Requires the pls package",
                         "Requires the mixOmics package"))
})

save(external_pls_benchmarks, 
     file = file.path("data", "external_pls_benchmarks.RData"), 
     compress = "xz", compression_level = 9)