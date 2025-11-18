## Bench helpers for performance-sensitive tests
## We use {bench} with warm-ups and a minimum timing window to avoid 0.000s.

make_dense_data <- function(n = 12000L, p = 512L, m = 8L, seed = 1L) {
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  B <- matrix(rnorm(p * m), p, m)
  Y <- scale(X, TRUE, FALSE) %*% B + 0.05 * matrix(rnorm(n * m), n, m)
  list(X = X, Y = Y)
}

make_bigmem_data <- function(n = 150000L, p = 64L, m = 4L, seed = 2L) {
  stopifnot(requireNamespace("bigmemory", quietly = TRUE))
  set.seed(seed)
  X <- matrix(rnorm(n * p), n, p)
  B <- matrix(rnorm(p * m), p, m)
  Y <- scale(X, TRUE, FALSE) %*% B + 0.05 * matrix(rnorm(n * m), n, m)
  list(Xbm = bigmemory::as.big.matrix(X),
       Ybm = bigmemory::as.big.matrix(Y))
}

# Warm-up a code path to prime BLAS/threads/caches.
warmup_dense_simpls <- function(dat) {
  withr::with_options(list(bigPLSR.simpls.solve = "tri"), {
    invisible(bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                               backend = "arma", algorithm = "simpls",
                               scores  = "r"))
  })
  withr::with_options(list(bigPLSR.simpls.solve = "solve"), {
    invisible(bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                               backend = "arma", algorithm = "simpls",
                               scores  = "r"))
  })
  invisible(NULL)
}

warmup_bigmem_simpls <- function(dat, chunk = 8192L) {
  withr::with_options(list(bigPLSR.simpls.stream_chunk = as.integer(chunk)), {
    invisible(bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                               backend = "bigmem", algorithm = "simpls",
                               scores  = "none"))
    invisible(bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                               backend = "bigmem", algorithm = "simpls",
                               scores  = "r"))
  })
  invisible(NULL)
}

# Bench a dense simpls call using {bench}; returns tibble with median etc.
bench_dense_simpls <- function(dat,
                               solver = c("tri", "solve"),
                               scores = c("r", "none"),
                               iterations = 8L,
                               min_time  = 1) {
  stopifnot(requireNamespace("bench", quietly = TRUE))
  solver <- match.arg(solver)
  scores <- match.arg(scores)
  warmup_dense_simpls(dat)
  bench::mark(
    run = withr::with_options(list(bigPLSR.simpls.solve = solver), {
      bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                       backend = "arma", algorithm = "simpls",
                       scores  = scores)
    }),
    iterations = iterations,
    min_time   = min_time,
    check      = FALSE
  )
}

# Bench a bigmem simpls call using {bench}; returns tibble with median etc.
bench_bigmem_simpls <- function(dat,
                                chunk = 8192L,
                                scores = c("none", "r"),
                                iterations = 6L,
                                min_time  = 1) {
  stopifnot(requireNamespace("bench", quietly = TRUE))
  scores <- match.arg(scores)
  warmup_bigmem_simpls(dat, chunk = chunk)
  bench::mark(
    run = withr::with_options(list(bigPLSR.simpls.stream_chunk = as.integer(chunk)), {
      bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                       backend = "bigmem", algorithm = "simpls",
                       scores  = scores)
    }),
    iterations = iterations,
    min_time   = min_time,
    check      = FALSE
  )
}

# Convenience to extract numeric median seconds from bench result
bench_median_sec <- function(bm) {
  # bm$median is a 'bench_time' difftime-like object; coerce to numeric seconds
  as.numeric(bm$median)
}

# Helpers to fetch medians when benchmarking multiple named expressions
bench_pick_median <- function(bm, name) {
  idx <- which(as.character(bm$expression) == name)
  stopifnot(length(idx) == 1L)
  as.numeric(bm$median[[idx]])
}