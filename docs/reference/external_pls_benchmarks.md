# Benchmark results against external PLS implementations

Pre-computed runtime comparisons between bigPLSR (dense and big.memory
backends) and reference implementations from the pls and mixOmics
packages.

## Usage

``` r
data(external_pls_benchmarks)
```

## Format

A data frame with 384 rows and 11 columns:

- task:

  Character vector identifying the task (`"pls1"` or `"pls2"`).

- algorithm:

  PLS algorithm used for the benchmark (e.g., `"simpls"`).

- package:

  Package providing the implementation.

- median_time_s:

  Median execution time in seconds.

- itr_per_sec:

  Iterations per second recorded by
  [`bench::mark()`](https://bench.r-lib.org/reference/mark.html).

- mem_alloc_bytes:

  Memory usage in bytes recorded by
  [`bench::mark()`](https://bench.r-lib.org/reference/mark.html).

- n:

  Number of observations in the simulated dataset.

- p:

  Number of predictors (X) in the simulated dataset.

- q:

  Number of responses (Y) in the simulated dataset.

- ncomp:

  Number of extracted components.

- notes:

  Helpful context on dependencies or configuration.

## Source

Generated via `inst/scripts/external_pls_benchmarks.R`.

## Details

Fix `task = "pls1"` and select algorithms in `"kernelpls"`, `"nipals"`
or `"simpls"` to get a full factorial design. Fix `task = "pls1"` and
fix `algorithm = "widekernelpls"` to get a full factorial design. Fix
`task = "pls2"` and select algorithms in `"kernelpls"`, `"nipals"` or
`"simpls"` to get a full factorial design. Fix `task = "pls2"` and fix
`algorithm = "widekernelpls"` to get a full factorial design.

## Examples

``` r
# \donttest{
data("external_pls_benchmarks", package = "bigPLSR")

sub_pls1 <- subset(external_pls_benchmarks, task == "pls1" & 
                                            algorithm != "widekernelpls")
sub_pls1$n     <- factor(sub_pls1$n)
sub_pls1$p     <- factor(sub_pls1$p)
sub_pls1$q     <- factor(sub_pls1$q)
sub_pls1$ncomp <- factor(sub_pls1$ncomp)
if (exists("replications")) replications(~ package + algorithm + task + n +
                                           p + ncomp, data = sub_pls1)
#> $package
#> [1] 24
#> 
#> $algorithm
#> [1] 32
#> 
#> $task
#> [1] 96
#> 
#> $n
#> [1] 48
#> 
#> $p
#> [1] 48
#> 
#> $ncomp
#> ncomp
#>   1   3  10 100 
#>  24  24  36  12 
#> 

sub_pls1_wide <- subset(external_pls_benchmarks, task == "pls1" & 
                                                 algorithm == "widekernelpls")
sub_pls1_wide$n     <- factor(sub_pls1_wide$n)
sub_pls1_wide$p     <- factor(sub_pls1_wide$p)
sub_pls1_wide$q     <- factor(sub_pls1_wide$q)
sub_pls1_wide$ncomp <- factor(sub_pls1_wide$ncomp)
if (exists("replications")) replications(~ package + algorithm + task + n + 
                                           p + ncomp, data = sub_pls1_wide)
#> $package
#> [1] 8
#> 
#> $algorithm
#> [1] 32
#> 
#> $task
#> [1] 32
#> 
#> $n
#> [1] 16
#> 
#> $p
#> [1] 16
#> 
#> $ncomp
#> ncomp
#>   1   3  10 100 
#>   8   8  12   4 
#> 

sub_pls2 <- subset(external_pls_benchmarks, task == "pls2" & 
                                            algorithm != "widekernelpls")
sub_pls2$n     <- factor(sub_pls2$n)
sub_pls2$p     <- factor(sub_pls2$p)
sub_pls2$q     <- factor(sub_pls2$q)
sub_pls2$ncomp <- factor(sub_pls2$ncomp)
if (exists("replications")) replications(~ package + algorithm + task + n + 
                                           p + ncomp, data = sub_pls2)
#> $package
#> [1] 48
#> 
#> $algorithm
#> [1] 64
#> 
#> $task
#> [1] 192
#> 
#> $n
#> [1] 96
#> 
#> $p
#> [1] 96
#> 
#> $ncomp
#> ncomp
#>  1  3 10 
#> 48 48 96 
#> 

sub_pls2_wide <- subset(external_pls_benchmarks, task == "pls2" & 
                                                 algorithm == "widekernelpls")
sub_pls2_wide$n     <- factor(sub_pls2_wide$n)
sub_pls2_wide$p     <- factor(sub_pls2_wide$p)
sub_pls2_wide$q     <- factor(sub_pls2_wide$q)
sub_pls2_wide$ncomp <- factor(sub_pls2_wide$ncomp)
if (exists("replications")) replications(~ package + algorithm + task + n + 
                                           p + ncomp, data = sub_pls2_wide)
#> $package
#> [1] 16
#> 
#> $algorithm
#> [1] 64
#> 
#> $task
#> [1] 64
#> 
#> $n
#> [1] 32
#> 
#> $p
#> [1] 32
#> 
#> $ncomp
#> ncomp
#>  1  3 10 
#> 16 16 32 
#> 
# }
```
