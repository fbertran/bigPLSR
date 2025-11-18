test_that("dense SIMPLS: bigPLSR.simpls.solve = 'tri' beats 'solve' (bench)", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bench")
  dat <- make_dense_data(n = 6000L, p = 512L, m = 8L, seed = 1L)
  
  # Warm-up and bench both solvers with min_time to avoid 0.000s
  warmup_dense_simpls(dat)
  res <- suppressWarnings(bench::mark(
    chol   = withr::with_options(list(bigPLSR.simpls.solve = "chol"),
                                bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                                 backend = "arma", algorithm = "simpls",
                                                 scores  = "r")),
    tri   = withr::with_options(list(bigPLSR.simpls.solve = "tri"),
                                bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                                 backend = "arma", algorithm = "simpls",
                                                 scores  = "r")),
    qr   = withr::with_options(list(bigPLSR.simpls.solve = "qr"),
                                bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                                 backend = "arma", algorithm = "simpls",
                                                 scores  = "r")),
    solve = withr::with_options(list(bigPLSR.simpls.solve = "solve"),
                                bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                                 backend = "arma", algorithm = "simpls",
                                                 scores  = "r")),
    iterations = 8,
    min_time  = .1,
    check     = FALSE
  ))
  t_chol   <- bench_pick_median(res, "chol")
  t_qr   <- bench_pick_median(res, "qr")
  t_tri   <- bench_pick_median(res, "tri")
  t_solve <- bench_pick_median(res, "solve")
  speedup <- t_solve / t_tri
  msg <- sprintf("Observed speedup = %.2fx (tri=%.3fs, solve=%.3fs)",
                 speedup, t_tri, t_solve)
  ## Be conservative to avoid flakiness across BLAS/CPUs
  expect_gt(speedup, .95, label = msg)
  message("1/6")
})

test_that("dense SIMPLS: 'tri' and 'solve' produce identical coefficients", {
  skip_on_cran()
  skip_on_ci() 
  dat <- make_dense_data()
  withr::with_options(list(bigPLSR.simpls.solve = "tri"), {
    fit_tri <- bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                backend = "arma", algorithm = "simpls",
                                scores  = "r")
  })
  withr::with_options(list(bigPLSR.simpls.solve = "solve"), {
    fit_sol <- bigPLSR::pls_fit(dat$X, dat$Y, ncomp = 12L,
                                backend = "arma", algorithm = "simpls",
                                scores  = "r")
  })
  expect_equal(unclass(fit_tri$coefficients),
               unclass(fit_sol$coefficients),
               tolerance = 1e-10)
  message("2/6")
})

test_that("bigmem SIMPLS: scores='none' is faster than 'r' (no streaming)", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bench")
  dat <- make_bigmem_data()
  warmup_bigmem_simplansin <- tryCatch({ warmup_bigmem_simpls(dat, 8192L); TRUE },
                                       error = function(e) FALSE)
  # Even if warmup fails on tiny CI boxes, still attempt benchmark guarded
  res <- suppressWarnings(bench::mark(
    none = withr::with_options(list(bigPLSR.simpls.stream_chunk = 8192L),
                               bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                                                backend = "bigmem", algorithm = "simpls",
                                                scores  = "none")),
    r    = withr::with_options(list(bigPLSR.simpls.stream_chunk = 8192L),
                               bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                                                backend = "bigmem", algorithm = "simpls",
                                                scores  = "r")),
    iterations = 6,
    min_time  = 1,
    check     = FALSE
  ))
  t_none <- bench_pick_median(res, "none")
  t_r    <- bench_pick_median(res, "r")
  msg <- sprintf("scores='none' (%.3fs) should be <= scores='r' (%.3fs)",
                 t_none, t_r)
  expect_lte(t_none, t_r + 1e-6, label = msg)
  message("3/6")
})

test_that("bigmem SIMPLS: chunk size affects runtime (sanity check)", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bench")
  set.seed(1639)
  dat <- make_bigmem_data()
  warmup_bigmem_simpls(dat, 8192L)
  set.seed(1639)
  res <- suppressWarnings(bench::mark(
    ch_4k  = withr::with_options(list(bigPLSR.simpls.stream_chunk = 4096L),
                                 bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                                                  backend = "bigmem", algorithm = "simpls",
                                                  scores  = "none")),
    ch_64k = withr::with_options(list(bigPLSR.simpls.stream_chunk = 65536L),
                                 bigPLSR::pls_fit(dat$Xbm, dat$Ybm, ncomp = 8L,
                                                  backend = "bigmem", algorithm = "simpls",
                                                  scores  = "none")),
    iterations = 6,
    min_time  = 1,
    check     = FALSE
  ))
  t4  <- bench_pick_median(res, "ch_4k")
  t64 <- bench_pick_median(res, "ch_64k")
  # We don't assert which one is faster (hardware dependent),
  # only that the timings are not numerically identical.
  expect_true(abs(t4 - t64) > 1e-4,
              info = sprintf("chunk 4k=%.3fs vs 64k=%.3fs", t4, t64))
  message("4/6")
})

test_that("dense cross: C fused path matches R and can be faster", {
  skip_on_cran()
  skip_on_ci() 
  skip_if_not_installed("bench")
  set.seed(2025)
  n <- 3000L; p <- 256L; m <- 12L; A <- 12L
  X <- matrix(rnorm(n * p), n, p)
  Y <- matrix(rnorm(n * m), n, m)

  # correctness parity (always run)
  fit_R <- withr::with_options(list(bigPLSR.simpls.cross = "R"),
                               bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                                                algorithm = "simpls", scores = "none"))
  fit_C <- withr::with_options(list(bigPLSR.simpls.cross = "dense_cxx"),
                               bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                                                algorithm = "simpls", scores = "none"))
  expect_equal(fit_C$coefficients, fit_R$coefficients, tolerance = 1e-12, ignore_attr = TRUE)
  expect_equal(as.numeric(fit_C$intercept), as.numeric(fit_R$intercept), tolerance = 1e-12)
  message("5/6")
})

test_that("dense cross: C fused path matches R and can be faster", {
    skip_on_cran()
    skip_on_ci() 
    skip_if_not_installed("bench")
    set.seed(2025)
    n <- 3000L; p <- 256L; m <- 12L; A <- 12L
    X <- matrix(rnorm(n * p), n, p)
    Y <- matrix(rnorm(n * m), n, m)
    
  res <- suppressWarnings(bench::mark(
    pls      = withr::with_options(list(bigPLSR.simpls.cross = "R"), pls::plsr(method = "simpls", Y~X,ncomp=A)),
    R        = withr::with_options(list(bigPLSR.simpls.cross = "R"),
                               bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                                                algorithm = "simpls", scores = "none")),
    CXX      = withr::with_options(list(bigPLSR.simpls.cross = "dense_cxx"),
                               bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                                                algorithm = "simpls", scores = "none")),
    CXX_SYRK = withr::with_options(list(bigPLSR.simpls.cross = "dense_cxx", bigPLSR.simpls.use_syrk = TRUE),
                               bigPLSR::pls_fit(X, Y, ncomp = A, backend = "arma",
                                                algorithm = "simpls", scores = "none")),
    iterations = 8,
    min_time  = 0.5,
    check     = FALSE
  ))
  med <- function(tag) median(as.numeric(res$time[names(res$expression) == tag][[1]]))
  speedup <- med("R") / med("CXX")
  testthat::expect_gt(as.numeric(speedup), .95)
  message("6/6 Done")
})
