test_that("stream block alignment respects option and clamps to n", {
  old <- getOption("bigPLSR.stream.block_align")
  on.exit(options(bigPLSR.stream.block_align = old), add = TRUE)
  
  options(bigPLSR.stream.block_align = 4096L)
  expect_equal(.bigPLSR_stream_block_size(n = 100000L, want = 5000L), 8192L)   # ceil to 4096 multiple
  expect_equal(.bigPLSR_stream_block_size(n = 7000L,   want = 5000L),  7000L)   # clamp to n
  expect_equal(.bigPLSR_stream_block_size(n = 3000L,   want = 0L),     3000L)   # at least align, but ≤ n
})