# Benchmarking PLS1 Implementations

``` r

library(bigPLSR)
library(bigmemory)
library(bench)
set.seed(123)
```

## Overview

The unified
[`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md)
interface now drives both the dense and streaming implementations of
single-response partial least squares regression. This vignette revisits
the benchmarking workflow with the modern API and introduces two
complementary perspectives:

1.  **Internal comparisons** that contrast the dense (in-memory) and
    streaming (big-memory) backends of
    [`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md).
2.  **External references** recorded against popular packages such as
    `pls` and `mixOmics`. These results are stored in the package to
    keep the vignette lightweight while still documenting performance
    relative to the wider ecosystem.

The chunks tagged with `eval = LOCAL` are only executed when the
environment variable `LOCAL` is set to `TRUE`, allowing CRAN checks to
skip the more time-consuming benchmarks.

## Simulated data

We create a synthetic regression problem with a modest latent structure
and keep both dense and big-memory versions of the predictors and
response so they can be reused in the benchmarking chunks.

Here is an example with `n=4000` en `p=50`

``` r

n <- 1500
p <- 80
ncomp <- 6

X <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
X[,] <- matrix(rnorm(n * p), nrow = n)

y_vec <- scale(X[,] %*% rnorm(p) + rnorm(n))

y <- bigmemory::big.matrix(nrow = n, ncol = 1, type = "double")
y[,] <- y_vec

X[1:6, 1:6]
#>             [,1]       [,2]        [,3]       [,4]       [,5]       [,6]
#> [1,] -0.56047565 -0.8209867 -0.15030748  0.8343715 -0.6992281  0.3500025
#> [2,] -0.23017749 -0.3072572 -0.32775713 -0.6984039  0.9964515  0.8144417
#> [3,]  1.55870831 -0.9020980 -1.44816529  1.3092405 -0.6927454 -0.5166661
#> [4,]  0.07050839  0.6270687 -0.69728458 -0.9801776 -0.1034830 -2.6922644
#> [5,]  0.12928774  1.1203550  2.59849023  0.7479851  0.6038661 -1.0969546
#> [6,]  1.71506499  2.1272136 -0.03741501  1.2577966 -0.6080450 -1.2554751
y[1:6,]
#> [1]  0.66723250  0.66189719 -0.77458416  0.07452428  0.28174414  0.15756565
```

## Internal benchmarks

The following chunk compares dense vs. streaming fits for both SIMPLS
and NIPALS. The dense backend receives base R matrices, while the
streaming backend consumes the `big.matrix` objects directly.

``` r

internal_bench <- bench::mark(
  dense_simpls = pls_fit(as.matrix(X[]), y_vec, ncomp = ncomp,
                         backend = "arma", algorithm = "simpls"),
  streaming_simpls = pls_fit(X, y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "simpls", chunk_size = 512L),
  dense_nipals = pls_fit(as.matrix(X[]), y_vec, ncomp = ncomp,
                         backend = "arma", algorithm = "nipals"),
  streaming_nipals = pls_fit(X, y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "nipals", chunk_size = 512L),
  dense_kernelpls = pls_fit(as.matrix(X[]), y_vec, ncomp = ncomp,
                         backend = "arma", algorithm = "kernelpls"),
  streaming_kernelpls = pls_fit(X, y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "kernelpls", chunk_size = 512L),
  dense_widekernelpls = pls_fit(as.matrix(X[]), y_vec, ncomp = ncomp,
                         backend = "arma", algorithm = "widekernelpls"),
  streaming_widekernelpls = pls_fit(X, y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "widekernelpls", chunk_size = 512L),
  iterations = 20,
  check = FALSE
)
internal_bench_res <-internal_bench[,2:5]
internal_bench_res <- as.matrix(internal_bench_res)
rownames(internal_bench_res) <- names(internal_bench$expression)
```

``` r

dotchart(internal_bench_res[,2], labels=rownames(internal_bench_res),xlab="median_time_s")
```

![](figures/benchmarking-pls1-internal-benchmark-plot-1.png)

``` r

dotchart(internal_bench_res[,3], labels=rownames(internal_bench_res),xlab="itr_per_sec")
```

![](figures/benchmarking-pls1-internal-benchmark-plot-2.png)

``` r

dotchart(internal_bench_res[,4], labels=rownames(internal_bench_res),xlab="mem_alloc_bytes")
```

![](figures/benchmarking-pls1-internal-benchmark-plot-3.png)

The results highlight the trade-off between throughput and memory usage:
SIMPLS shines on dense matrices, whereas the streaming backend scales to
larger-than-memory inputs thanks to block processing.

## External references

To avoid heavy dependencies at build time we ship a pre-computed
benchmark dataset that contrasts `bigPLSR` with implementations from the
`pls` and `mixOmics` packages. The dataset was generated with the helper
script stored in `inst/scripts/external_pls_benchmarks.R`.

``` r

data("external_pls_benchmarks", package = "bigPLSR")
sub_pls1 <- subset(external_pls_benchmarks,task=="pls1" & !algorithm=="widekernelpls")
sub_pls1$n <- factor(sub_pls1$n)
sub_pls1$p <- factor(sub_pls1$p)
sub_pls1$q <- factor(sub_pls1$q)
sub_pls1$ncomp <- factor(sub_pls1$ncomp)
replications(~package+algorithm+task+n+p+ncomp,data=sub_pls1)
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

sub_pls1_wide <- subset(external_pls_benchmarks,external_pls_benchmarks$task=="pls1" & algorithm=="widekernelpls")
sub_pls1_wide$n <- factor(sub_pls1_wide$n)
sub_pls1_wide$p <- factor(sub_pls1_wide$p)
sub_pls1_wide$q <- factor(sub_pls1_wide$q)
sub_pls1_wide$ncomp <- factor(sub_pls1_wide$ncomp)
replications(~package+algorithm+task+n+p+ncomp,data=sub_pls1_wide)
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

sub_pls2 <- subset(external_pls_benchmarks,external_pls_benchmarks$task=="pls2" & !algorithm=="widekernelpls")
sub_pls2$n <- factor(sub_pls2$n)
sub_pls2$p <- factor(sub_pls2$p)
sub_pls2$q <- factor(sub_pls2$q)
sub_pls2$ncomp <- factor(sub_pls2$ncomp)
replications(~package+algorithm+task+n+p+ncomp,data=sub_pls2)
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

sub_pls2_wide <- subset(external_pls_benchmarks,external_pls_benchmarks$task=="pls2" & algorithm=="widekernelpls")
sub_pls2_wide$n <- factor(sub_pls2_wide$n)
sub_pls2_wide$p <- factor(sub_pls2_wide$p)
sub_pls2_wide$q <- factor(sub_pls2_wide$q)
sub_pls2_wide$ncomp <- factor(sub_pls2_wide$ncomp)
replications(~package+algorithm+task+n+p+ncomp,data=sub_pls2_wide)
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
```

``` r

sub_pls1
#>     task algorithm            package median_time_s  itr_per_sec
#> 1   pls1    simpls      bigPLSR_dense  0.0060451836  165.1562621
#> 2   pls1    simpls bigPLSR_big.memory  0.0035867620  275.9417591
#> 3   pls1    simpls                pls  0.0020867360  469.9431634
#> 4   pls1    simpls           mixOmics  0.0044788605  223.4486896
#> 5   pls1 kernelpls      bigPLSR_dense  0.0441235030   22.6636584
#> 6   pls1 kernelpls bigPLSR_big.memory  0.0005215200 1733.6289782
#> 7   pls1 kernelpls                pls  0.0021023160  475.5957998
#> 8   pls1 kernelpls           mixOmics  0.0044129530  224.0172938
#> 13  pls1    nipals      bigPLSR_dense  0.0007704311 1208.7122974
#> 14  pls1    nipals bigPLSR_big.memory  0.0064723010  154.6317423
#> 15  pls1    nipals                pls  0.0023130355  423.5614303
#> 16  pls1    nipals           mixOmics  0.0045207831  221.5220634
#> 17  pls1    simpls      bigPLSR_dense  0.0061238625  160.2659625
#> 18  pls1    simpls bigPLSR_big.memory  0.0034934051  280.9346862
#> 19  pls1    simpls                pls  0.0026201049  378.4570550
#> 20  pls1    simpls           mixOmics  0.0081250110  122.0214629
#> 21  pls1 kernelpls      bigPLSR_dense  0.0459685031   21.7108564
#> 22  pls1 kernelpls bigPLSR_big.memory  0.0011338961  885.9666822
#> 23  pls1 kernelpls                pls  0.0026236720  383.4893703
#> 24  pls1 kernelpls           mixOmics  0.0081293570  121.1046267
#> 29  pls1    nipals      bigPLSR_dense  0.0017690270  544.4038157
#> 30  pls1    nipals bigPLSR_big.memory  0.0063486451  155.6316749
#> 31  pls1    nipals                pls  0.0032869700  308.6738445
#> 32  pls1    nipals           mixOmics  0.0082580560  121.5590611
#> 33  pls1    simpls      bigPLSR_dense  0.0062094295  160.3209005
#> 34  pls1    simpls bigPLSR_big.memory  0.0035923790  275.3568879
#> 35  pls1    simpls                pls  0.0046357060  207.3664119
#> 36  pls1    simpls           mixOmics  0.0209913850   47.6021094
#> 37  pls1 kernelpls      bigPLSR_dense  0.0498161890   20.0737957
#> 38  pls1 kernelpls bigPLSR_big.memory  0.0032459290  301.8471240
#> 39  pls1 kernelpls                pls  0.0047320150  211.2860747
#> 40  pls1 kernelpls           mixOmics  0.0209948905   47.6736952
#> 45  pls1    nipals      bigPLSR_dense  0.0047139545  211.8381858
#> 46  pls1    nipals bigPLSR_big.memory  0.0065105130  153.5810813
#> 47  pls1    nipals                pls  0.0074865180  132.9719814
#> 48  pls1    nipals           mixOmics  0.0210122335   46.8847723
#> 145 pls1    simpls      bigPLSR_dense  6.5033901315    0.1547340
#> 146 pls1    simpls bigPLSR_big.memory  3.5608631200    0.2812449
#> 147 pls1    simpls                pls  0.3611949530    2.9793809
#> 148 pls1    simpls           mixOmics  0.4746793860    2.0961402
#> 149 pls1 kernelpls      bigPLSR_dense  0.0888563070   11.0521410
#> 150 pls1 kernelpls bigPLSR_big.memory  0.0596174030   16.5969947
#> 151 pls1 kernelpls                pls  0.3368680336    3.1522969
#> 152 pls1 kernelpls           mixOmics  0.4710328870    2.2186854
#> 157 pls1    nipals      bigPLSR_dense  0.0818347905   12.0507715
#> 158 pls1    nipals bigPLSR_big.memory  6.2263474941    0.1599221
#> 159 pls1    nipals                pls  0.1974032740    5.0479061
#> 160 pls1    nipals           mixOmics  0.2946449011    3.3824261
#> 161 pls1    simpls      bigPLSR_dense  6.1359636295    0.1626558
#> 162 pls1    simpls bigPLSR_big.memory  3.3408962340    0.2994953
#> 163 pls1    simpls                pls  0.2738218415    3.2636860
#> 164 pls1    simpls           mixOmics  0.7325085420    1.4078995
#> 165 pls1 kernelpls      bigPLSR_dense  0.2181904176    4.6082604
#> 166 pls1 kernelpls bigPLSR_big.memory  0.1305059521    7.7058833
#> 167 pls1 kernelpls                pls  0.3852149671    2.6743926
#> 168 pls1 kernelpls           mixOmics  0.7455238690    1.2945480
#> 173 pls1    nipals      bigPLSR_dense  0.1886118900    5.1953353
#> 174 pls1    nipals bigPLSR_big.memory  6.2367988861    0.1602238
#> 175 pls1    nipals                pls  0.3296010500    3.0281963
#> 176 pls1    nipals           mixOmics  0.5922050660    1.7045469
#> 177 pls1    simpls      bigPLSR_dense  6.2095294170    0.1609821
#> 178 pls1    simpls bigPLSR_big.memory  3.4807222855    0.2865378
#> 179 pls1    simpls                pls  0.4312646910    2.2350247
#> 180 pls1    simpls           mixOmics  1.7723871550    0.5695185
#> 181 pls1 kernelpls      bigPLSR_dense  0.6455497560    1.5651878
#> 182 pls1 kernelpls bigPLSR_big.memory  0.3514678876    2.7636579
#> 183 pls1 kernelpls                pls  0.5371060476    1.8759850
#> 184 pls1 kernelpls           mixOmics  1.8362041675    0.5476303
#> 189 pls1    nipals      bigPLSR_dense  0.5208903835    1.9089468
#> 190 pls1    nipals bigPLSR_big.memory  6.2764627576    0.1592679
#> 191 pls1    nipals                pls  0.7559157085    1.3152253
#> 192 pls1    nipals           mixOmics  1.6296500625    0.6067950
#> 289 pls1    simpls      bigPLSR_dense  0.0062788220  158.3886973
#> 290 pls1    simpls bigPLSR_big.memory  0.0041523160  235.3499445
#> 291 pls1    simpls                pls  0.0049248995  202.1917049
#> 292 pls1    simpls           mixOmics  0.0229660476   42.7499866
#> 293 pls1 kernelpls      bigPLSR_dense  0.0506123269   19.4606804
#> 294 pls1 kernelpls bigPLSR_big.memory  0.0033179866  299.1089816
#> 295 pls1 kernelpls                pls  0.0049014065  202.4541097
#> 296 pls1 kernelpls           mixOmics  0.0221207505   45.1716313
#> 301 pls1    nipals      bigPLSR_dense  0.0047306620  210.7978021
#> 302 pls1    nipals bigPLSR_big.memory  0.0066085235  148.0432944
#> 303 pls1    nipals                pls  0.0079110935  126.1707804
#> 304 pls1    nipals           mixOmics  0.0220015431   45.5108450
#> 337 pls1    simpls      bigPLSR_dense  6.1752819530    0.1613928
#> 338 pls1    simpls bigPLSR_big.memory  3.3039307570    0.3007239
#> 339 pls1    simpls                pls  0.5349956956    1.9629343
#> 340 pls1    simpls           mixOmics  1.6342236126    0.6042950
#> 341 pls1 kernelpls      bigPLSR_dense  0.6289640670    1.5716088
#> 342 pls1 kernelpls bigPLSR_big.memory  0.3496842645    2.7791228
#> 343 pls1 kernelpls                pls  0.5294607366    1.9623568
#> 344 pls1 kernelpls           mixOmics  1.6767431134    0.5890119
#> 349 pls1    nipals      bigPLSR_dense  0.5095970975    1.9054646
#> 350 pls1    nipals bigPLSR_big.memory  6.2690384570    0.1591941
#> 351 pls1    nipals                pls  0.7049397730    1.3617341
#> 352 pls1    nipals           mixOmics  1.5486491710    0.6395732
#>     mem_alloc_bytes     n    p q ncomp
#> 1            104720  1000  100 1     1
#> 2            895872  1000  100 1     1
#> 3           7479024  1000  100 1     1
#> 4           7550384  1000  100 1     1
#> 5          42461256  1000  100 1     1
#> 6            821328  1000  100 1     1
#> 7           7462504  1000  100 1     1
#> 8           7550384  1000  100 1     1
#> 13            14928  1000  100 1     1
#> 14           830224  1000  100 1     1
#> 15          8265304  1000  100 1     1
#> 16          7550384  1000  100 1     1
#> 17           107920  1000  100 1     3
#> 18           899072  1000  100 1     3
#> 19          7764576  1000  100 1     3
#> 20         12407296  1000  100 1     3
#> 21         42480456  1000  100 1     3
#> 22           840528  1000  100 1     3
#> 23          7720248  1000  100 1     3
#> 24         12304368  1000  100 1     3
#> 29            18128  1000  100 1     3
#> 30           865424  1000  100 1     3
#> 31         10122568  1000  100 1     3
#> 32         12304368  1000  100 1     3
#> 33           119120  1000  100 1    10
#> 34           910272  1000  100 1    10
#> 35          9013040  1000  100 1    10
#> 36         28992768  1000  100 1    10
#> 37         42547656  1000  100 1    10
#> 38           907728  1000  100 1    10
#> 39          8959992  1000  100 1    10
#> 40         28992768  1000  100 1    10
#> 45            29328  1000  100 1    10
#> 46           988624  1000  100 1    10
#> 47         16876368  1000  100 1    10
#> 48         28992768  1000  100 1    10
#> 145         8226320 10000 1000 1     1
#> 146        88138272 10000 1000 1     1
#> 147       722792304 10000 1000 1     1
#> 148       651508960 10000 1000 1     1
#> 149          282544 10000 1000 1     1
#> 150        80194128 10000 1000 1     1
#> 151       722624584 10000 1000 1     1
#> 152       651508960 10000 1000 1     1
#> 157          130128 10000 1000 1     1
#> 158        80282224 10000 1000 1     1
#> 159       802652584 10000 1000 1     1
#> 160       651508960 10000 1000 1     1
#> 161         8258320 10000 1000 1     3
#> 162        88170272 10000 1000 1     3
#> 163       725626656 10000 1000 1     3
#> 164       986912144 10000 1000 1     3
#> 165          474544 10000 1000 1     3
#> 166        80386128 10000 1000 1     3
#> 167       725179128 10000 1000 1     3
#> 168       986912144 10000 1000 1     3
#> 173          162128 10000 1000 1     3
#> 174        80634224 10000 1000 1     3
#> 175       965206648 10000 1000 1     3
#> 176       986912144 10000 1000 1     3
#> 177         8370320 10000 1000 1    10
#> 178        88282272 10000 1000 1    10
#> 179       738038720 10000 1000 1    10
#> 180      2161313744 10000 1000 1    10
#> 181         1146544 10000 1000 1    10
#> 182        81058128 10000 1000 1    10
#> 183       737488872 10000 1000 1    10
#> 184      2161313744 10000 1000 1    10
#> 189          274128 10000 1000 1    10
#> 190        81866224 10000 1000 1    10
#> 191      1536642048 10000 1000 1    10
#> 192      2161313744 10000 1000 1    10
#> 289          119120  1000  100 1    10
#> 290          910272  1000  100 1    10
#> 291         9013040  1000  100 1    10
#> 292        28992768  1000  100 1    10
#> 293        42547656  1000  100 1    10
#> 294          907728  1000  100 1    10
#> 295         8959992  1000  100 1    10
#> 296        28992768  1000  100 1    10
#> 301           29328  1000  100 1    10
#> 302          988624  1000  100 1    10
#> 303        16876368  1000  100 1    10
#> 304        28992768  1000  100 1    10
#> 337         8370320 10000 1000 1   100
#> 338        88282272 10000 1000 1   100
#> 339       738038720 10000 1000 1   100
#> 340      2161313744 10000 1000 1   100
#> 341         1146544 10000 1000 1   100
#> 342        81058128 10000 1000 1   100
#> 343       737488872 10000 1000 1   100
#> 344      2161313744 10000 1000 1   100
#> 349          274128 10000 1000 1   100
#> 350        81866224 10000 1000 1   100
#> 351      1536642048 10000 1000 1   100
#> 352      2161313744 10000 1000 1   100
#>                                         notes
#> 1        Run via pls_fit() with dense backend
#> 2   Run via pls_fit() with big.memory backend
#> 3                    Requires the pls package
#> 4               Requires the mixOmics package
#> 5        Run via pls_fit() with dense backend
#> 6   Run via pls_fit() with big.memory backend
#> 7                    Requires the pls package
#> 8               Requires the mixOmics package
#> 13       Run via pls_fit() with dense backend
#> 14  Run via pls_fit() with big.memory backend
#> 15                   Requires the pls package
#> 16              Requires the mixOmics package
#> 17       Run via pls_fit() with dense backend
#> 18  Run via pls_fit() with big.memory backend
#> 19                   Requires the pls package
#> 20              Requires the mixOmics package
#> 21       Run via pls_fit() with dense backend
#> 22  Run via pls_fit() with big.memory backend
#> 23                   Requires the pls package
#> 24              Requires the mixOmics package
#> 29       Run via pls_fit() with dense backend
#> 30  Run via pls_fit() with big.memory backend
#> 31                   Requires the pls package
#> 32              Requires the mixOmics package
#> 33       Run via pls_fit() with dense backend
#> 34  Run via pls_fit() with big.memory backend
#> 35                   Requires the pls package
#> 36              Requires the mixOmics package
#> 37       Run via pls_fit() with dense backend
#> 38  Run via pls_fit() with big.memory backend
#> 39                   Requires the pls package
#> 40              Requires the mixOmics package
#> 45       Run via pls_fit() with dense backend
#> 46  Run via pls_fit() with big.memory backend
#> 47                   Requires the pls package
#> 48              Requires the mixOmics package
#> 145      Run via pls_fit() with dense backend
#> 146 Run via pls_fit() with big.memory backend
#> 147                  Requires the pls package
#> 148             Requires the mixOmics package
#> 149      Run via pls_fit() with dense backend
#> 150 Run via pls_fit() with big.memory backend
#> 151                  Requires the pls package
#> 152             Requires the mixOmics package
#> 157      Run via pls_fit() with dense backend
#> 158 Run via pls_fit() with big.memory backend
#> 159                  Requires the pls package
#> 160             Requires the mixOmics package
#> 161      Run via pls_fit() with dense backend
#> 162 Run via pls_fit() with big.memory backend
#> 163                  Requires the pls package
#> 164             Requires the mixOmics package
#> 165      Run via pls_fit() with dense backend
#> 166 Run via pls_fit() with big.memory backend
#> 167                  Requires the pls package
#> 168             Requires the mixOmics package
#> 173      Run via pls_fit() with dense backend
#> 174 Run via pls_fit() with big.memory backend
#> 175                  Requires the pls package
#> 176             Requires the mixOmics package
#> 177      Run via pls_fit() with dense backend
#> 178 Run via pls_fit() with big.memory backend
#> 179                  Requires the pls package
#> 180             Requires the mixOmics package
#> 181      Run via pls_fit() with dense backend
#> 182 Run via pls_fit() with big.memory backend
#> 183                  Requires the pls package
#> 184             Requires the mixOmics package
#> 189      Run via pls_fit() with dense backend
#> 190 Run via pls_fit() with big.memory backend
#> 191                  Requires the pls package
#> 192             Requires the mixOmics package
#> 289      Run via pls_fit() with dense backend
#> 290 Run via pls_fit() with big.memory backend
#> 291                  Requires the pls package
#> 292             Requires the mixOmics package
#> 293      Run via pls_fit() with dense backend
#> 294 Run via pls_fit() with big.memory backend
#> 295                  Requires the pls package
#> 296             Requires the mixOmics package
#> 301      Run via pls_fit() with dense backend
#> 302 Run via pls_fit() with big.memory backend
#> 303                  Requires the pls package
#> 304             Requires the mixOmics package
#> 337      Run via pls_fit() with dense backend
#> 338 Run via pls_fit() with big.memory backend
#> 339                  Requires the pls package
#> 340             Requires the mixOmics package
#> 341      Run via pls_fit() with dense backend
#> 342 Run via pls_fit() with big.memory backend
#> 343                  Requires the pls package
#> 344             Requires the mixOmics package
#> 349      Run via pls_fit() with dense backend
#> 350 Run via pls_fit() with big.memory backend
#> 351                  Requires the pls package
#> 352             Requires the mixOmics package
```

The table reports median execution times (in seconds), the number of
iterations and memory use per second for a representative
single-response scenario. The notes column indicates the additional
packages required to reproduce those measurements.

## Takeaways

Dense vs streaming backends. On small/medium data that fits in RAM,
in-memory implementations (e.g., pls) are typically fastest (median
≈0.36 s for SIMPLS in our runs). However, they materialize large
cross-products/Gram matrices and memory grows as O(p^2) (or O(n^2) in
kernel views). In contrast, bigPLSR’s streaming big-memory backend keeps
memory bounded via chunked BLAS and never forms those intermediates. In
our PLS2 benchmark, streaming used ~7–8× less RAM than pls (≈89 MB vs
≈732 MB median) while remaining competitive in runtime (≈3.5 s vs 0.36
s). PLS1 shows the same pattern: streaming is often fast enough while
dramatically reducing memory. As n or p grow, the streaming backend
scales where dense approaches become memory-limited.
