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
#' `data/external_pls_benchmarks.RData`.

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

run_benchmark <- function(task_name, algorithm, n, p, q, ncomp) {
  if(task_name == "pls1"){
    dat <- make_task(n, p, q)
    dat$Y <- dat$Y[,1,drop=FALSE]
    Xbm <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
    Xbm[,] <- dat$X
    Ybm <- bigmemory::big.matrix(nrow = n, ncol = 1, type = "double")
    Ybm[,] <- dat$Y
  res <- bench::mark(
    bigPLSR_dense = pls_fit(dat$X, dat$Y, ncomp = min(10, ncomp), algorithm = algorithm),
    bigPLSR_big.memory = pls_fit(Xbm, Ybm, ncomp = min(10, ncomp), algorithm = algorithm),
    pls = {
      if (q == 1) {
        pls::plsr(dat$Y ~ dat$X, ncomp = min(10, ncomp), method = if(algorithm=="nipals"){"oscorespls"} else {algorithm})
      } else {
        pls::plsr(dat$Y ~ dat$X, ncomp = min(10, ncomp), method = if(algorithm=="nipals"){"oscorespls"} else {algorithm})
      }
    },
    mixOmics = {
      if (q == 1) {
        mixOmics::pls(dat$X, dat$Y[,1, drop = FALSE], ncomp = min(10, ncomp))
      } else {
        mixOmics::spls(dat$X, dat$Y, ncomp = min(10, ncomp))
      }
    },
    iterations = 20,
    check = FALSE
  )} else {if(task_name == "pls2"){
    dat <- make_task(n, p, q)
    Xbm <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
    Xbm[,] <- dat$X
    Ybm <- bigmemory::big.matrix(nrow = n, ncol = q, type = "double")
    Ybm[,] <- dat$Y
    res <- bench::mark(
      bigPLSR_dense = pls_fit(dat$X, dat$Y, ncomp = min(10, ncomp), algorithm = algorithm),
      bigPLSR_big.memory = pls_fit(Xbm, Ybm, ncomp = min(10, ncomp), algorithm = algorithm),
      pls = {
        if (q == 1) {
          pls::plsr(dat$Y ~ dat$X, ncomp = min(10, ncomp), method = if(algorithm=="nipals"){"oscorespls"} else {algorithm})
        } else {
          pls::plsr(dat$Y ~ dat$X, ncomp = min(10, ncomp), method = if(algorithm=="nipals"){"oscorespls"} else {algorithm})
        }
      },
      mixOmics = {
        if (q == 1) {
          mixOmics::pls(dat$X, dat$Y[,1, drop = FALSE], ncomp = min(10, ncomp))
        } else {
          mixOmics::spls(dat$X, dat$Y, ncomp = min(10, ncomp))
        }
      },
      iterations = 20,
      check = FALSE
    )  
  }}
  cat(paste(task_name, algorithm, n, p, q, ncomp, "\n", sep="_"))
  data.frame(
    task = task_name,
    algorithm = algorithm,
    package = names(res$expression),
    median_time_s = as.numeric(res$median),
    itr_per_sec = res$`itr/sec`,
    mem_alloc_bytes = as.numeric(res$mem_alloc),
    n = n,
    p = p,
    q = q,
    ncomp = ncomp,
    stringsAsFactors = FALSE
  )
}


benchmarks_pls1 <- rbind(
  run_benchmark("pls1", algorithm = "simpls", n = 1000, p = 100, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "kernelpls", n = 1000, p = 100, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 100, p = 5000, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "nipals", n = 1000, p = 100, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "simpls", n = 1000, p = 100, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "kernelpls", n = 1000, p = 100, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 100, p = 5000, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "nipals", n = 1000, p = 100, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "simpls", n = 1000, p = 100, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "kernelpls", n = 1000, p = 100, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 100, p = 5000, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "nipals", n = 1000, p = 100, q = 1, ncomp = 10)
)

benchmarks_pls2 <- rbind(
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 10, ncomp = 100)
)

benchmarks_pls2_bigq <- rbind(
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 100, ncomp = 1),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 100, ncomp = 1),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 100, ncomp = 1),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 100, ncomp = 1),
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 100, ncomp = 3),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 100, ncomp = 3),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 100, ncomp = 3),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 100, ncomp = 3),
  run_benchmark("pls2", algorithm = "simpls", n = 1000, p = 100, q = 100, ncomp = 10),
  run_benchmark("pls2", algorithm = "kernelpls", n = 1000, p = 100, q = 100, ncomp = 10),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 100, p = 5000, q = 100, ncomp = 10),
  run_benchmark("pls2", algorithm = "nipals", n = 1000, p = 100, q = 100, ncomp = 100)
)

benchmarks_pls1_bign_bigp <- rbind(
  run_benchmark("pls1", algorithm = "simpls", n = 10000, p = 1000, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "kernelpls", n = 10000, p = 1000, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 1000, p = 50000, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "nipals", n = 10000, p = 1000, q = 1, ncomp = 1),
  run_benchmark("pls1", algorithm = "simpls", n = 10000, p = 1000, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "kernelpls", n = 10000, p = 1000, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 1000, p = 50000, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "nipals", n = 10000, p = 1000, q = 1, ncomp = 3),
  run_benchmark("pls1", algorithm = "simpls", n = 10000, p = 1000, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "kernelpls", n = 10000, p = 1000, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "widekernelpls", n = 1000, p = 50000, q = 1, ncomp = 10),
  run_benchmark("pls1", algorithm = "nipals", n = 10000, p = 1000, q = 1, ncomp = 10)
)

benchmarks_pls2_bign_bigp <- rbind(
  run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 10, ncomp = 1),
  run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 10, ncomp = 3),
  run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 10, ncomp = 10),
  run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 10, ncomp = 10)
)

benchmarks_pl2_bign_bigp_bigq <- rbind(
run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 100, ncomp = 1),
run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 100, ncomp = 1),
run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 100, ncomp = 1),
run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 100, ncomp = 1),
run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 100, ncomp = 3),
run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 100, ncomp = 3),
run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 100, ncomp = 3),
run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 100, ncomp = 3),
run_benchmark("pls2", algorithm = "simpls", n = 10000, p = 1000, q = 100, ncomp = 10),
run_benchmark("pls2", algorithm = "kernelpls", n = 10000, p = 1000, q = 100, ncomp = 10),
run_benchmark("pls2", algorithm = "widekernelpls", n = 1000, p = 50000, q = 100, ncomp = 10),
run_benchmark("pls2", algorithm = "nipals", n = 10000, p = 1000, q = 100, ncomp = 10)
)

benchmarks_all <- rbind(benchmarks_pls1,
                        benchmarks_pls2,
                        benchmarks_pls2_bigq)

benchmarks_all <- rbind(benchmarks_pls1,
                        benchmarks_pls2,
                        benchmarks_pls2_bigq,
                        benchmarks_pls1_bign_bigp,
                        benchmarks_pls2_bign_bigp,
                        benchmarks_pl2_bign_bigp_bigq)

external_pls_benchmarks <- within(benchmarks_all, {
  notes <- ifelse(package == "bigPLSR_dense",
                  "Run via pls_fit() with dense backend",
                  ifelse(package == "bigPLSR_big.memory",
                  "Run via pls_fit() with big.memory backend",
                  ifelse(package == "pls",
                         "Requires the pls package",
                         "Requires the mixOmics package")))
})

save(external_pls_benchmarks, 
     file = file.path("data", "external_pls_benchmarks.RData"), 
     compress = "xz", compression_level = 9)
