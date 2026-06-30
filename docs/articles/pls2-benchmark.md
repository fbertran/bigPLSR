# Benchmarking PLS2 Implementations

``` r

library(bigPLSR)
library(bigmemory)
library(bench)
set.seed(456)
```

## Overview

The package offers dense (`pls2_dense`) and streaming (`pls2_stream`)
solvers for multi-response partial least squares regression (PLS2). This
vignette demonstrates how to benchmark both variants on a synthetic
dataset featuring three correlated response variables.

### Recent additions

Beyond the dense and streaming SIMPLS/NIPALS solvers, bigPLSR now ships
with Kalman-filter PLS (`algorithm = "kf_pls"`), double-RKHS modelling
(`algorithm = "rkhs_xy"`) and optional coefficient thresholding. The
resampling helpers
([`pls_cross_validate()`](https://fbertran.github.io/bigPLSR/reference/pls_cross_validate.md),
[`pls_bootstrap()`](https://fbertran.github.io/bigPLSR/reference/pls_bootstrap.md))
can also leverage the [`future`](https://future.futureverse.org)
ecosystem for parallel execution.

To benchmark these variants simply change the `algorithm` parameter in
the chunks below, for example:

``` r

bench::mark(
  dense = pls_fit(X[], Y_mat, ncomp = ncomp, algorithm = "rkhs_xy"),
  streaming = pls_fit(X, Y, ncomp = ncomp, backend = "bigmem",
                      algorithm = "kf_pls", chunk_size = 1024L)
)
```

and remember to reset your `future` plan after enabling parallelism:

``` r

future::plan(future::multisession, workers = 2)
pls_cross_validate(X[], Y_mat, ncomp = 4, folds = 3,
                   parallel = TRUE)
future::plan(future::sequential)
```

Multi-response benchmarks follow the same principles as the PLS1 case.
We focus on the
[`pls_fit()`](https://fbertran.github.io/bigPLSR/reference/pls_fit.md)
API and contrast its dense and streaming backends before reporting the
stored results against third-party packages.

## Simulated data

``` r

n <- 1200
p <- 60
q <- 3
ncomp <- 4

X <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
X[,] <- matrix(rnorm(n * p), nrow = n)

loading_matrix <- matrix(rnorm(p * q), nrow = p)
latent_scores <- matrix(rnorm(n * q), nrow = n)
Y_mat <- scale(latent_scores %*% t(loading_matrix[1:q, , drop = FALSE]) +
                 matrix(rnorm(n * q, sd = 0.5), nrow = n))

Y <- bigmemory::big.matrix(nrow = n, ncol = q, type = "double")
Y[,] <- Y_mat

X[1:6, 1:6]
#>            [,1]         [,2]        [,3]       [,4]        [,5]       [,6]
#> [1,] -1.3435214 -0.348899457  0.70772263  1.1760806  0.05196595  0.6838164
#> [2,]  0.6217756  1.068279438 -1.17880479 -0.8208425 -1.50933535 -0.8963854
#> [3,]  0.8008747 -0.005793261 -0.04600936 -0.8817557  0.17372573 -1.2315819
#> [4,] -1.3888924  0.560411440 -1.15682882 -1.1538845 -1.26526869 -0.3837956
#> [5,] -0.7143569  2.533318058 -1.47324626  0.2554595 -0.25199313  1.1685923
#> [6,] -0.3240611  0.436737176  0.48330946  0.2118484  0.47779676  1.1944133
Y[1:6, 1:min(6, q)]
#>            [,1]         [,2]        [,3]
#> [1,]  1.8176535 -0.431537115 -0.24548255
#> [2,]  1.1617501 -1.573377901 -1.21506233
#> [3,] -0.4016607  0.412473967 -0.03420375
#> [4,] -0.9068912  1.328097496  0.67450957
#> [5,]  1.2561929 -0.839498415 -1.71712784
#> [6,]  0.5802884 -0.006450224  1.41649712
```

## Internal benchmarks

``` r

internal_bench <- bench::mark(
  dense_simpls = pls_fit(as.matrix(X[]), Y_mat, ncomp = ncomp,
                         backend = "arma", algorithm = "simpls"),
  streaming_simpls = pls_fit(X, Y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "simpls", chunk_size = 512L),
  dense_nipals = pls_fit(as.matrix(X[]), Y_mat, ncomp = ncomp,
                         backend = "arma", algorithm = "nipals"),
  streaming_nipals = pls_fit(X, Y, ncomp = ncomp, backend = "bigmem",
                             algorithm = "nipals", chunk_size = 512L),
  iterations = 15,
  check = FALSE
)
internal_bench
#> # A tibble: 4 × 6
#>   expression            min   median `itr/sec` mem_alloc `gc/sec`
#>   <bch:expr>       <bch:tm> <bch:tm>     <dbl> <bch:byt>    <dbl>
#> 1 dense_simpls          3ms   3.06ms     326.     2.04MB        0
#> 2 streaming_simpls   1.66ms   1.68ms     592.   628.94KB        0
#> 3 dense_nipals       9.86ms   9.96ms     100.   574.56KB        0
#> 4 streaming_nipals  27.85ms  28.03ms      34.0  658.29KB        0
```

The dense path again excels when memory allows, whereas the streaming
backend prioritises scalability via block-wise processing.

## External references

``` r

data("external_pls_benchmarks", package = "bigPLSR")
subset(external_pls_benchmarks, task == "pls2")
#>     task     algorithm            package median_time_s  itr_per_sec
#> 49  pls2        simpls      bigPLSR_dense  7.120081e-03 1.399652e+02
#> 50  pls2        simpls bigPLSR_big.memory  3.834689e-03 2.545964e+02
#> 51  pls2        simpls                pls  3.482007e-03 2.860761e+02
#> 52  pls2        simpls           mixOmics  5.855128e-03 1.691964e+02
#> 53  pls2     kernelpls      bigPLSR_dense  5.264746e-02 1.899056e+01
#> 54  pls2     kernelpls bigPLSR_big.memory  5.652055e-04 1.701649e+03
#> 55  pls2     kernelpls                pls  3.451585e-03 2.885003e+02
#> 56  pls2     kernelpls           mixOmics  5.765769e-03 1.732900e+02
#> 57  pls2 widekernelpls      bigPLSR_dense  4.605411e-02 2.171396e+01
#> 58  pls2 widekernelpls bigPLSR_big.memory  2.186694e-03 4.521202e+02
#> 59  pls2 widekernelpls                pls  2.413531e-02 4.138397e+01
#> 60  pls2 widekernelpls           mixOmics  1.908312e-02 5.244135e+01
#> 61  pls2        nipals      bigPLSR_dense  1.175341e-02 8.437194e+01
#> 62  pls2        nipals bigPLSR_big.memory  2.194955e-02 4.561655e+01
#> 63  pls2        nipals                pls  2.618026e-02 3.827063e+01
#> 64  pls2        nipals           mixOmics  5.808839e-03 1.719635e+02
#> 65  pls2        simpls      bigPLSR_dense  7.172253e-03 1.380721e+02
#> 66  pls2        simpls bigPLSR_big.memory  3.827104e-03 2.563459e+02
#> 67  pls2        simpls                pls  4.212176e-03 2.369476e+02
#> 68  pls2        simpls           mixOmics  1.178188e-02 8.491185e+01
#> 69  pls2     kernelpls      bigPLSR_dense  6.769330e-02 1.466727e+01
#> 70  pls2     kernelpls bigPLSR_big.memory  1.182112e-03 8.162977e+02
#> 71  pls2     kernelpls                pls  4.161131e-03 2.391603e+02
#> 72  pls2     kernelpls           mixOmics  1.177631e-02 8.404231e+01
#> 73  pls2 widekernelpls      bigPLSR_dense  8.934355e-02 1.118655e+01
#> 74  pls2 widekernelpls bigPLSR_big.memory  5.242937e-03 1.866432e+02
#> 75  pls2 widekernelpls                pls  2.966153e-02 3.379981e+01
#> 76  pls2 widekernelpls           mixOmics  3.831171e-02 2.610168e+01
#> 77  pls2        nipals      bigPLSR_dense  4.063383e-02 2.456427e+01
#> 78  pls2        nipals bigPLSR_big.memory  1.075419e-01 9.301410e+00
#> 79  pls2        nipals                pls  7.387882e-02 1.353426e+01
#> 80  pls2        nipals           mixOmics  1.172631e-02 8.565988e+01
#> 81  pls2        simpls      bigPLSR_dense  7.425654e-03 1.346546e+02
#> 82  pls2        simpls bigPLSR_big.memory  4.044650e-03 2.431415e+02
#> 83  pls2        simpls                pls  7.231929e-03 1.378009e+02
#> 84  pls2        simpls           mixOmics  3.237143e-02 3.072704e+01
#> 85  pls2     kernelpls      bigPLSR_dense  1.224882e-01 8.164049e+00
#> 86  pls2     kernelpls bigPLSR_big.memory  3.494451e-03 2.748492e+02
#> 87  pls2     kernelpls                pls  7.494862e-03 1.348653e+02
#> 88  pls2     kernelpls           mixOmics  3.303632e-02 3.030338e+01
#> 89  pls2 widekernelpls      bigPLSR_dense  2.394804e-01 4.144136e+00
#> 90  pls2 widekernelpls bigPLSR_big.memory  1.648811e-02 5.921678e+01
#> 91  pls2 widekernelpls                pls  6.210118e-02 1.105851e+01
#> 92  pls2 widekernelpls           mixOmics  1.246962e-01 6.579412e+00
#> 93  pls2        nipals      bigPLSR_dense  9.577266e-02 1.042355e+01
#> 94  pls2        nipals bigPLSR_big.memory  4.760183e-01 2.098655e+00
#> 95  pls2        nipals                pls  1.941141e-01 5.150694e+00
#> 96  pls2        nipals           mixOmics  3.304953e-02 3.008813e+01
#> 97  pls2        simpls      bigPLSR_dense  1.932603e-02 5.127135e+01
#> 98  pls2        simpls bigPLSR_big.memory  9.015121e-03 1.101052e+02
#> 99  pls2        simpls                pls  1.770130e-02 5.637404e+01
#> 100 pls2        simpls           mixOmics  1.900584e-02 5.264451e+01
#> 101 pls2     kernelpls      bigPLSR_dense  6.007267e-02 1.663928e+01
#> 102 pls2     kernelpls bigPLSR_big.memory  8.541325e-04 1.121615e+03
#> 103 pls2     kernelpls                pls  1.775359e-02 5.651641e+01
#> 104 pls2     kernelpls           mixOmics  1.887583e-02 5.293168e+01
#> 105 pls2 widekernelpls      bigPLSR_dense  4.742109e-02 2.099435e+01
#> 106 pls2 widekernelpls bigPLSR_big.memory  3.370958e-03 2.934597e+02
#> 107 pls2 widekernelpls                pls  2.681298e-02 3.779052e+01
#> 108 pls2 widekernelpls           mixOmics  7.449075e-02 1.343417e+01
#> 109 pls2        nipals      bigPLSR_dense  3.911978e-02 2.555092e+01
#> 110 pls2        nipals bigPLSR_big.memory  7.650871e-02 1.305760e+01
#> 111 pls2        nipals                pls  4.743122e-02 2.107277e+01
#> 112 pls2        nipals           mixOmics  1.906176e-02 5.255502e+01
#> 113 pls2        simpls      bigPLSR_dense  2.247368e-02 4.445638e+01
#> 114 pls2        simpls bigPLSR_big.memory  1.244026e-02 8.001525e+01
#> 115 pls2        simpls                pls  2.384585e-02 4.201748e+01
#> 116 pls2        simpls           mixOmics  4.853586e-02 2.053421e+01
#> 117 pls2     kernelpls      bigPLSR_dense  9.047814e-02 1.105505e+01
#> 118 pls2     kernelpls bigPLSR_big.memory  1.861646e-03 5.104304e+02
#> 119 pls2     kernelpls                pls  2.322777e-02 4.280268e+01
#> 120 pls2     kernelpls           mixOmics  4.834396e-02 2.067876e+01
#> 121 pls2 widekernelpls      bigPLSR_dense  1.031736e-01 9.566646e+00
#> 122 pls2 widekernelpls bigPLSR_big.memory  7.027215e-03 1.260660e+02
#> 123 pls2 widekernelpls                pls  4.710053e-02 1.101579e+01
#> 124 pls2 widekernelpls           mixOmics  2.111830e-01 4.402695e+00
#> 125 pls2        nipals      bigPLSR_dense  2.074080e-01 4.814438e+00
#> 126 pls2        nipals bigPLSR_big.memory  6.856785e-01 1.458201e+00
#> 127 pls2        nipals                pls  1.362973e-01 7.336902e+00
#> 128 pls2        nipals           mixOmics  4.847024e-02 2.067250e+01
#> 129 pls2        simpls      bigPLSR_dense  3.730465e-02 2.668616e+01
#> 130 pls2        simpls bigPLSR_big.memory  2.433875e-02 4.067987e+01
#> 131 pls2        simpls                pls  4.679924e-02 2.136787e+01
#> 132 pls2        simpls           mixOmics  1.503539e-01 6.650973e+00
#> 133 pls2     kernelpls      bigPLSR_dense  1.967330e-01 5.078581e+00
#> 134 pls2     kernelpls bigPLSR_big.memory  5.323809e-03 1.806102e+02
#> 135 pls2     kernelpls                pls  4.377998e-02 2.284149e+01
#> 136 pls2     kernelpls           mixOmics  1.492201e-01 6.528839e+00
#> 137 pls2 widekernelpls      bigPLSR_dense  2.908055e-01 3.362013e+00
#> 138 pls2 widekernelpls bigPLSR_big.memory  1.946467e-02 5.008337e+01
#> 139 pls2 widekernelpls                pls  1.026065e-01 7.440373e+00
#> 140 pls2 widekernelpls           mixOmics  6.646347e-01 1.486103e+00
#> 141 pls2        nipals      bigPLSR_dense  8.330493e-01 1.196314e+00
#> 142 pls2        nipals bigPLSR_big.memory  4.391516e+00 2.276423e-01
#> 143 pls2        nipals                pls  4.477764e-01 2.233258e+00
#> 144 pls2        nipals           mixOmics  1.505480e-01 6.646532e+00
#> 193 pls2        simpls      bigPLSR_dense  6.309363e+00 1.585065e-01
#> 194 pls2        simpls bigPLSR_big.memory  3.497791e+00 2.859247e-01
#> 195 pls2        simpls                pls  3.419946e-01 2.749000e+00
#> 196 pls2        simpls           mixOmics  4.503820e-01 2.097276e+00
#> 197 pls2     kernelpls      bigPLSR_dense  9.432889e-01 1.040672e+00
#> 198 pls2     kernelpls bigPLSR_big.memory  5.854103e-02 1.643222e+01
#> 199 pls2     kernelpls                pls  3.422486e-01 2.649781e+00
#> 200 pls2     kernelpls           mixOmics  4.515143e-01 2.104715e+00
#> 201 pls2 widekernelpls      bigPLSR_dense  2.145082e+01 4.673179e-02
#> 202 pls2 widekernelpls bigPLSR_big.memory  3.493717e-01 2.815957e+00
#> 203 pls2 widekernelpls                pls  1.942898e+01 5.145024e-02
#> 204 pls2 widekernelpls           mixOmics  2.179235e+00 4.541845e-01
#> 205 pls2        nipals      bigPLSR_dense  2.068101e+00 4.517067e-01
#> 206 pls2        nipals bigPLSR_big.memory  6.296180e+00 1.591039e-01
#> 207 pls2        nipals                pls  2.578546e+00 3.911680e-01
#> 208 pls2        nipals           mixOmics  4.128616e-01 2.414049e+00
#> 209 pls2        simpls      bigPLSR_dense  6.275229e+00 1.587181e-01
#> 210 pls2        simpls bigPLSR_big.memory  3.500080e+00 2.857462e-01
#> 211 pls2        simpls                pls  3.950639e-01 2.405834e+00
#> 212 pls2        simpls           mixOmics  1.071424e+00 9.621914e-01
#> 213 pls2     kernelpls      bigPLSR_dense  2.457075e+00 4.057651e-01
#> 214 pls2     kernelpls bigPLSR_big.memory  1.253336e-01 7.782986e+00
#> 215 pls2     kernelpls                pls  4.999669e-01 2.188557e+00
#> 216 pls2     kernelpls           mixOmics  1.088277e+00 8.969374e-01
#> 217 pls2 widekernelpls      bigPLSR_dense  2.853491e+01 3.500674e-02
#> 218 pls2 widekernelpls bigPLSR_big.memory  8.872444e-01 1.100254e+00
#> 219 pls2 widekernelpls                pls  2.353821e+01 4.244972e-02
#> 220 pls2 widekernelpls           mixOmics  4.682576e+00 2.131930e-01
#> 221 pls2        nipals      bigPLSR_dense  3.676615e+00 2.591175e-01
#> 222 pls2        nipals bigPLSR_big.memory  1.409305e+01 7.101126e-02
#> 223 pls2        nipals                pls  6.665440e+00 1.498888e-01
#> 224 pls2        nipals           mixOmics  9.350531e-01 1.047136e+00
#> 225 pls2        simpls      bigPLSR_dense  6.298712e+00 1.588801e-01
#> 226 pls2        simpls bigPLSR_big.memory  3.413522e+00 2.918573e-01
#> 227 pls2        simpls                pls  6.889283e-01 1.476556e+00
#> 228 pls2        simpls           mixOmics  3.093444e+00 3.257282e-01
#> 229 pls2     kernelpls      bigPLSR_dense  7.988876e+00 1.244374e-01
#> 230 pls2     kernelpls bigPLSR_big.memory  3.680395e-01 2.636549e+00
#> 231 pls2     kernelpls                pls  6.871629e-01 1.562790e+00
#> 232 pls2     kernelpls           mixOmics  2.895363e+00 3.455009e-01
#> 233 pls2 widekernelpls      bigPLSR_dense  4.852656e+01 2.059706e-02
#> 234 pls2 widekernelpls bigPLSR_big.memory  2.731539e+00 3.655164e-01
#> 235 pls2 widekernelpls                pls  3.777036e+01 2.667407e-02
#> 236 pls2 widekernelpls           mixOmics  1.300218e+01 7.683072e-02
#> 237 pls2        nipals      bigPLSR_dense  3.606770e+01 2.770877e-02
#> 238 pls2        nipals bigPLSR_big.memory  1.341817e+02 7.466282e-03
#> 239 pls2        nipals                pls  2.028668e+01 4.936161e-02
#> 240 pls2        nipals           mixOmics  2.634110e+00 3.787394e-01
#> 241 pls2        simpls      bigPLSR_dense  7.326091e+00 1.360943e-01
#> 242 pls2        simpls bigPLSR_big.memory  3.616469e+00 2.765349e-01
#> 243 pls2        simpls                pls  1.421763e+00 7.033523e-01
#> 244 pls2        simpls           mixOmics  1.531064e+00 6.531404e-01
#> 245 pls2     kernelpls      bigPLSR_dense  9.506041e-01 1.051029e+00
#> 246 pls2     kernelpls bigPLSR_big.memory  6.050341e-02 1.642222e+01
#> 247 pls2     kernelpls                pls  1.540560e+00 6.486713e-01
#> 248 pls2     kernelpls           mixOmics  1.637565e+00 6.105833e-01
#> 249 pls2 widekernelpls      bigPLSR_dense  2.182893e+01 4.539441e-02
#> 250 pls2 widekernelpls bigPLSR_big.memory  3.665587e-01 2.673948e+00
#> 251 pls2 widekernelpls                pls  1.972794e+01 5.021136e-02
#> 252 pls2 widekernelpls           mixOmics  7.708876e+00 1.283845e-01
#> 253 pls2        nipals      bigPLSR_dense  2.752244e+00 3.290323e-01
#> 254 pls2        nipals bigPLSR_big.memory  9.417478e+00 1.063625e-01
#> 255 pls2        nipals                pls  2.813499e+00 3.555340e-01
#> 256 pls2        nipals           mixOmics  1.580067e+00 6.334909e-01
#> 257 pls2        simpls      bigPLSR_dense  7.455953e+00 1.335826e-01
#> 258 pls2        simpls bigPLSR_big.memory  3.887993e+00 2.573655e-01
#> 259 pls2        simpls                pls  1.580360e+00 6.228051e-01
#> 260 pls2        simpls           mixOmics  4.498919e+00 2.220574e-01
#> 261 pls2     kernelpls      bigPLSR_dense  2.906596e+00 3.341038e-01
#> 262 pls2     kernelpls bigPLSR_big.memory  1.382854e-01 6.562282e+00
#> 263 pls2     kernelpls                pls  1.703039e+00 5.854046e-01
#> 264 pls2     kernelpls           mixOmics  4.512791e+00 2.195872e-01
#> 265 pls2 widekernelpls      bigPLSR_dense  2.963823e+01 3.379320e-02
#> 266 pls2 widekernelpls bigPLSR_big.memory  9.002348e-01 1.110139e+00
#> 267 pls2 widekernelpls                pls  2.384161e+01 4.258298e-02
#> 268 pls2 widekernelpls           mixOmics  2.081519e+01 4.767220e-02
#> 269 pls2        nipals      bigPLSR_dense  1.408437e+01 6.962744e-02
#> 270 pls2        nipals bigPLSR_big.memory  5.705946e+01 1.759331e-02
#> 271 pls2        nipals                pls  8.023360e+00 1.249047e-01
#> 272 pls2        nipals           mixOmics  4.382419e+00 2.249435e-01
#> 273 pls2        simpls      bigPLSR_dense  7.525037e+00 1.328067e-01
#> 274 pls2        simpls bigPLSR_big.memory  3.913807e+00 2.557014e-01
#> 275 pls2        simpls                pls  2.022675e+00 4.826692e-01
#> 276 pls2        simpls           mixOmics  1.420802e+01 7.038199e-02
#> 277 pls2     kernelpls      bigPLSR_dense  9.697582e+00 1.012366e-01
#> 278 pls2     kernelpls bigPLSR_big.memory  3.819721e-01 2.568275e+00
#> 279 pls2     kernelpls                pls  2.058282e+00 4.740146e-01
#> 280 pls2     kernelpls           mixOmics  1.422527e+01 6.968572e-02
#> 281 pls2 widekernelpls      bigPLSR_dense  5.718163e+01 1.737826e-02
#> 282 pls2 widekernelpls bigPLSR_big.memory  2.825720e+00 3.564109e-01
#> 283 pls2 widekernelpls                pls  3.734700e+01 2.669714e-02
#> 284 pls2 widekernelpls           mixOmics  6.888371e+01 1.451777e-02
#> 285 pls2        nipals      bigPLSR_dense  7.541323e+01 1.294568e-02
#> 286 pls2        nipals bigPLSR_big.memory  3.811755e+02 2.620968e-03
#> 287 pls2        nipals                pls  2.640921e+01 3.776758e-02
#> 288 pls2        nipals           mixOmics  1.420909e+01 7.007476e-02
#> 305 pls2        simpls      bigPLSR_dense  7.497711e-03 1.333907e+02
#> 306 pls2        simpls bigPLSR_big.memory  4.155473e-03 2.388086e+02
#> 307 pls2        simpls                pls  7.653163e-03 1.319696e+02
#> 308 pls2        simpls           mixOmics  3.350266e-02 2.994403e+01
#> 309 pls2     kernelpls      bigPLSR_dense  1.242683e-01 8.035064e+00
#> 310 pls2     kernelpls bigPLSR_big.memory  3.464889e-03 2.877941e+02
#> 311 pls2     kernelpls                pls  7.359172e-03 1.355891e+02
#> 312 pls2     kernelpls           mixOmics  3.397613e-02 2.944409e+01
#> 313 pls2 widekernelpls      bigPLSR_dense  2.407071e-01 4.129654e+00
#> 314 pls2 widekernelpls bigPLSR_big.memory  1.716645e-02 5.751937e+01
#> 315 pls2 widekernelpls                pls  5.536624e-02 1.785374e+01
#> 316 pls2 widekernelpls           mixOmics  1.108436e-01 8.954716e+00
#> 317 pls2        nipals      bigPLSR_dense  1.190147e-01 8.344615e+00
#> 318 pls2        nipals bigPLSR_big.memory  6.515927e-01 1.534309e+00
#> 319 pls2        nipals                pls  2.003472e-01 4.978335e+00
#> 320 pls2        nipals           mixOmics  3.375095e-02 2.951392e+01
#> 321 pls2        simpls      bigPLSR_dense  3.683102e-02 2.691890e+01
#> 322 pls2        simpls bigPLSR_big.memory  2.474278e-02 3.998584e+01
#> 323 pls2        simpls                pls  4.763642e-02 2.110398e+01
#> 324 pls2        simpls           mixOmics  1.520066e-01 6.573756e+00
#> 325 pls2     kernelpls      bigPLSR_dense  1.968019e-01 5.079041e+00
#> 326 pls2     kernelpls bigPLSR_big.memory  5.401176e-03 1.829141e+02
#> 327 pls2     kernelpls                pls  4.563501e-02 2.201507e+01
#> 328 pls2     kernelpls           mixOmics  1.532541e-01 6.513871e+00
#> 329 pls2 widekernelpls      bigPLSR_dense  3.034882e-01 3.284866e+00
#> 330 pls2 widekernelpls bigPLSR_big.memory  2.010958e-02 4.794885e+01
#> 331 pls2 widekernelpls                pls  9.123222e-02 1.081412e+01
#> 332 pls2 widekernelpls           mixOmics  6.927910e-01 1.437205e+00
#> 333 pls2        nipals      bigPLSR_dense  8.576115e-01 1.165103e+00
#> 334 pls2        nipals bigPLSR_big.memory  4.315265e+00 2.308993e-01
#> 335 pls2        nipals                pls  4.494351e-01 2.225018e+00
#> 336 pls2        nipals           mixOmics  1.543894e-01 6.475478e+00
#> 353 pls2        simpls      bigPLSR_dense  6.329638e+00 1.578499e-01
#> 354 pls2        simpls bigPLSR_big.memory  3.523030e+00 2.837708e-01
#> 355 pls2        simpls                pls  5.680624e-01 1.728639e+00
#> 356 pls2        simpls           mixOmics  2.844275e+00 3.526984e-01
#> 357 pls2     kernelpls      bigPLSR_dense  8.436175e+00 1.187776e-01
#> 358 pls2     kernelpls bigPLSR_big.memory  3.815881e-01 2.571431e+00
#> 359 pls2     kernelpls                pls  7.294291e-01 1.406790e+00
#> 360 pls2     kernelpls           mixOmics  3.055918e+00 3.267786e-01
#> 361 pls2 widekernelpls      bigPLSR_dense  5.111861e+01 1.950671e-02
#> 362 pls2 widekernelpls bigPLSR_big.memory  2.746516e+00 3.614256e-01
#> 363 pls2 widekernelpls                pls  3.718312e+01 2.682237e-02
#> 364 pls2 widekernelpls           mixOmics  1.316891e+01 7.595419e-02
#> 365 pls2        nipals      bigPLSR_dense  1.014015e+01 9.799419e-02
#> 366 pls2        nipals bigPLSR_big.memory  6.524409e+01 1.531731e-02
#> 367 pls2        nipals                pls  2.042867e+01 4.888707e-02
#> 368 pls2        nipals           mixOmics  2.742070e+00 3.629917e-01
#> 369 pls2        simpls      bigPLSR_dense  7.573426e+00 1.319950e-01
#> 370 pls2        simpls bigPLSR_big.memory  3.838224e+00 2.600652e-01
#> 371 pls2        simpls                pls  1.906750e+00 5.205236e-01
#> 372 pls2        simpls           mixOmics  1.425391e+01 7.011841e-02
#> 373 pls2     kernelpls      bigPLSR_dense  9.608967e+00 1.033116e-01
#> 374 pls2     kernelpls bigPLSR_big.memory  3.821283e-01 2.533252e+00
#> 375 pls2     kernelpls                pls  2.043786e+00 4.870992e-01
#> 376 pls2     kernelpls           mixOmics  1.440603e+01 6.912951e-02
#> 377 pls2 widekernelpls      bigPLSR_dense  5.716155e+01 1.748790e-02
#> 378 pls2 widekernelpls bigPLSR_big.memory  2.786712e+00 3.590855e-01
#> 379 pls2 widekernelpls                pls  3.827468e+01 2.613465e-02
#> 380 pls2 widekernelpls           mixOmics  6.871450e+01 1.454585e-02
#> 381 pls2        nipals      bigPLSR_dense  8.504559e+01 1.114233e-02
#> 382 pls2        nipals bigPLSR_big.memory  4.164587e+02 2.398284e-03
#> 383 pls2        nipals                pls  2.660601e+01 3.759870e-02
#> 384 pls2        nipals           mixOmics  1.427050e+01 6.975301e-02
#>     mem_alloc_bytes     n     p   q ncomp
#> 49           103024  1000   100  10     1
#> 50           982272  1000   100  10     1
#> 51          8479304  1000   100  10     1
#> 52          8318896  1000   100  10     1
#> 53         42452360  1000   100  10     1
#> 54           900528  1000   100  10     1
#> 55          8436448  1000   100  10     1
#> 56          8190272  1000   100  10     1
#> 57         12968760   100  5000  10     1
#> 58          4530928   100  5000  10     1
#> 59         35414384   100  5000  10     1
#> 60         34578992   100  5000  10     1
#> 61            14080  1000   100  10     1
#> 62           915560  1000   100  10     1
#> 63         13434968  1000   100  10     1
#> 64          8190272  1000   100  10     1
#> 65           106512  1000   100  10     3
#> 66           985760  1000   100  10     3
#> 67          9444440  1000   100  10     3
#> 68         13388032  1000   100  10     3
#> 69         42471848  1000   100  10     3
#> 70           920016  1000   100  10     3
#> 71          9402656  1000   100  10     3
#> 72         13388032  1000   100  10     3
#> 73         13130648   100  5000  10     3
#> 74          4692816   100  5000  10     3
#> 75         39041656   100  5000  10     3
#> 76         54606752   100  5000  10     3
#> 77            17568  1000   100  10     3
#> 78           931360  1000   100  10     3
#> 79         24566632  1000   100  10     3
#> 80         13388032  1000   100  10     3
#> 81           118272  1000   100  10    10
#> 82           997520  1000   100  10    10
#> 83         12918736  1000   100  10    10
#> 84         31625968  1000   100  10    10
#> 85         42539608  1000   100  10    10
#> 86           987776  1000   100  10    10
#> 87         12874168  1000   100  10    10
#> 88         31625968  1000   100  10    10
#> 89         13696808   100  5000  10    10
#> 90          5258976   100  5000  10    10
#> 91         53342272   100  5000  10    10
#> 92        127150688   100  5000  10    10
#> 93            29328  1000   100  10    10
#> 94          1004720  1000   100  10    10
#> 95         54886008  1000   100  10    10
#> 96         31625968  1000   100  10    10
#> 97           252960  1000   100 100     1
#> 98          1851360  1000   100 100     1
#> 99         18594944  1000   100 100     1
#> 100        14977592  1000   100 100     1
#> 101        42528600  1000   100 100     1
#> 102         1695072  1000   100 100     1
#> 103        18575032  1000   100 100     1
#> 104        14800240  1000   100 100     1
#> 105        16573000   100  5000 100     1
#> 106         8205472   100  5000 100     1
#> 107        43570672   100  5000 100     1
#> 108        38867640   100  5000 100     1
#> 109           90320  1000   100 100     1
#> 110         1778512  1000   100 100     1
#> 111        23817480  1000   100 100     1
#> 112        14800240  1000   100 100     1
#> 113          257760  1000   100 100     3
#> 114         1856160  1000   100 100     3
#> 115        27460048  1000   100 100     3
#> 116        24555680  1000   100 100     3
#> 117        42549400  1000   100 100     3
#> 118         1715872  1000   100 100     3
#> 119        27405544  1000   100 100     3
#> 120        24555680  1000   100 100     3
#> 121        16736200   100  5000 100     3
#> 122         8368672   100  5000 100     3
#> 123        62765600   100  5000 100     3
#> 124        66586440   100  5000 100     3
#> 125           95120  1000   100 100     3
#> 126         1800912  1000   100 100     3
#> 127        44869208  1000   100 100     3
#> 128        24555680  1000   100 100     3
#> 129          274560  1000   100 100    10
#> 130         1872960  1000   100 100    10
#> 131        56015344  1000   100 100    10
#> 132        58798008  1000   100 100    10
#> 133        42622200  1000   100 100    10
#> 134         1788672  1000   100 100    10
#> 135        55928376  1000   100 100    10
#> 136        58798008  1000   100 100    10
#> 137        17307400   100  5000 100    10
#> 138         8939872   100  5000 100    10
#> 139       131220112   100  5000 100    10
#> 140       166101528   100  5000 100    10
#> 141          111920  1000   100 100    10
#> 142         1879312  1000   100 100    10
#> 143       116080232  1000   100 100    10
#> 144        58798008  1000   100 100    10
#> 193         8210224 10000  1000  10     1
#> 194        89002272 10000  1000  10     1
#> 195       732448600 10000  1000  10     1
#> 196       657916048 10000  1000  10     1
#> 197          194448 10000  1000  10     1
#> 198        80986128 10000  1000  10     1
#> 199       732288928 10000  1000  10     1
#> 200       657916048 10000  1000  10     1
#> 201      1245640176  1000 50000  10     1
#> 202       405290128  1000 50000  10     1
#> 203      3305468960  1000 50000  10     1
#> 204      3225391600  1000 50000  10     1
#> 205          122080 10000  1000  10     1
#> 206        81082272 10000  1000  10     1
#> 207       854535432 10000  1000  10     1
#> 208       657916048 10000  1000  10     1
#> 209         8242512 10000  1000  10     3
#> 210        89034560 10000  1000  10     3
#> 211       742210520 10000  1000  10     3
#> 212       997737408 10000  1000  10     3
#> 213          386736 10000  1000  10     3
#> 214        81178416 10000  1000  10     3
#> 215       741787136 10000  1000  10     3
#> 216       997737408 10000  1000  10     3
#> 217      1247256464  1000 50000  10     3
#> 218       406906416  1000 50000  10     3
#> 219      3443148112  1000 50000  10     3
#> 220      4865512960  1000 50000  10     3
#> 221          154368 10000  1000  10     3
#> 222        81290560 10000  1000  10     3
#> 223      1097484040 10000  1000  10     3
#> 224       997737408 10000  1000  10     3
#> 225         8355072 10000  1000  10    10
#> 226        89147120 10000  1000  10    10
#> 227       776353216 10000  1000  10    10
#> 228      2187598944 10000  1000  10    10
#> 229         1059296 10000  1000  10    10
#> 230        81850976 10000  1000  10    10
#> 231       775883848 10000  1000  10    10
#> 232      2187598944 10000  1000  10    10
#> 233      1252913024  1000 50000  10    10
#> 234       412562976  1000 50000  10    10
#> 235      3941532384  1000 50000  10    10
#> 236     10630434496  1000 50000  10    10
#> 237          266928 10000  1000  10    10
#> 238        82019120 10000  1000  10    10
#> 239      1938583272 10000  1000  10    10
#> 240      2187598944 10000  1000  10    10
#> 241         9656160 10000  1000 100     1
#> 242        97647360 10000  1000 100     1
#> 243       829340832 10000  1000 100     1
#> 244       723476696 10000  1000 100     1
#> 245          918688 10000  1000 100     1
#> 246        88908672 10000  1000 100     1
#> 247       829181160 10000  1000 100     1
#> 248       723476696 10000  1000 100     1
#> 249      1281644416  1000 50000 100     1
#> 250       442012672  1000 50000 100     1
#> 251      3387267744  1000 50000 100     1
#> 252      3267912248  1000 50000 100     1
#> 253          846320 10000  1000 100     1
#> 254        89726512 10000  1000 100     1
#> 255       956883960 10000  1000 100     1
#> 256       723476696 10000  1000 100     1
#> 257         9689760 10000  1000 100     3
#> 258        97680960 10000  1000 100     3
#> 259       909018352 10000  1000 100     3
#> 260      1107997096 10000  1000 100     3
#> 261         1112288 10000  1000 100     3
#> 262        89102272 10000  1000 100     3
#> 263       908594968 10000  1000 100     3
#> 264      1107997096 10000  1000 100     3
#> 265      1283262016  1000 50000 100     3
#> 266       443630272  1000 50000 100     3
#> 267      3680010256  1000 50000 100     3
#> 268      4984412648  1000 50000 100     3
#> 269          879920 10000  1000 100     3
#> 270        89936112 10000  1000 100     3
#> 271      1309070888 10000  1000 100     3
#> 272      1107997096 10000  1000 100     3
#> 273         9807360 10000  1000 100    10
#> 274        97798560 10000  1000 100    10
#> 275      1162711904 10000  1000 100    10
#> 276      2454357784 10000  1000 100    10
#> 277         1789888 10000  1000 100    10
#> 278        89779872 10000  1000 100    10
#> 279      1162242536 10000  1000 100    10
#> 280      2454357784 10000  1000 100    10
#> 281      1288923616  1000 50000 100    10
#> 282       449291872  1000 50000 100    10
#> 283      4714714672  1000 50000 100    10
#> 284     11016713336  1000 50000 100    10
#> 285          997520 10000  1000 100    10
#> 286        90669712 10000  1000 100    10
#> 287      2516549912 10000  1000 100    10
#> 288      2454357784 10000  1000 100    10
#> 305          118272  1000   100  10    10
#> 306          997520  1000   100  10    10
#> 307        12918736  1000   100  10    10
#> 308        31625968  1000   100  10    10
#> 309        42539608  1000   100  10    10
#> 310          987776  1000   100  10    10
#> 311        12874168  1000   100  10    10
#> 312        31625968  1000   100  10    10
#> 313        13696808   100  5000  10    10
#> 314         5258976   100  5000  10    10
#> 315        53417792   100  5000  10    10
#> 316       127150688   100  5000  10    10
#> 317           29328  1000   100  10    10
#> 318         1004720  1000   100  10    10
#> 319        55313416  1000   100  10    10
#> 320        31625968  1000   100  10    10
#> 321          274560  1000   100 100    10
#> 322         1872960  1000   100 100    10
#> 323        56015344  1000   100 100    10
#> 324        58798008  1000   100 100    10
#> 325        42622200  1000   100 100    10
#> 326         1788672  1000   100 100    10
#> 327        55928376  1000   100 100    10
#> 328        58798008  1000   100 100    10
#> 329        17307400   100  5000 100    10
#> 330         8939872   100  5000 100    10
#> 331       131224400   100  5000 100    10
#> 332       166101528   100  5000 100    10
#> 333          111920  1000   100 100    10
#> 334         1879312  1000   100 100    10
#> 335       116080232  1000   100 100    10
#> 336        58798008  1000   100 100    10
#> 353         8355072 10000  1000  10    10
#> 354        89147120 10000  1000  10    10
#> 355       776353216 10000  1000  10    10
#> 356      2187598944 10000  1000  10    10
#> 357         1059296 10000  1000  10    10
#> 358        81850976 10000  1000  10    10
#> 359       775883848 10000  1000  10    10
#> 360      2187598944 10000  1000  10    10
#> 361      1252913024  1000 50000  10    10
#> 362       412562976  1000 50000  10    10
#> 363      3939826208  1000 50000  10    10
#> 364     10630434496  1000 50000  10    10
#> 365          266928 10000  1000  10    10
#> 366        82019120 10000  1000  10    10
#> 367      1938503224 10000  1000  10    10
#> 368      2187598944 10000  1000  10    10
#> 369         9807360 10000  1000 100    10
#> 370        97798560 10000  1000 100    10
#> 371      1162711904 10000  1000 100    10
#> 372      2454357784 10000  1000 100    10
#> 373         1789888 10000  1000 100    10
#> 374        89779872 10000  1000 100    10
#> 375      1162242536 10000  1000 100    10
#> 376      2454357784 10000  1000 100    10
#> 377      1288923616  1000 50000 100    10
#> 378       449291872  1000 50000 100    10
#> 379      4714714672  1000 50000 100    10
#> 380     11016713336  1000 50000 100    10
#> 381          997520 10000  1000 100    10
#> 382        90669712 10000  1000 100    10
#> 383      2516549912 10000  1000 100    10
#> 384      2454357784 10000  1000 100    10
#>                                         notes
#> 49       Run via pls_fit() with dense backend
#> 50  Run via pls_fit() with big.memory backend
#> 51                   Requires the pls package
#> 52              Requires the mixOmics package
#> 53       Run via pls_fit() with dense backend
#> 54  Run via pls_fit() with big.memory backend
#> 55                   Requires the pls package
#> 56              Requires the mixOmics package
#> 57       Run via pls_fit() with dense backend
#> 58  Run via pls_fit() with big.memory backend
#> 59                   Requires the pls package
#> 60              Requires the mixOmics package
#> 61       Run via pls_fit() with dense backend
#> 62  Run via pls_fit() with big.memory backend
#> 63                   Requires the pls package
#> 64              Requires the mixOmics package
#> 65       Run via pls_fit() with dense backend
#> 66  Run via pls_fit() with big.memory backend
#> 67                   Requires the pls package
#> 68              Requires the mixOmics package
#> 69       Run via pls_fit() with dense backend
#> 70  Run via pls_fit() with big.memory backend
#> 71                   Requires the pls package
#> 72              Requires the mixOmics package
#> 73       Run via pls_fit() with dense backend
#> 74  Run via pls_fit() with big.memory backend
#> 75                   Requires the pls package
#> 76              Requires the mixOmics package
#> 77       Run via pls_fit() with dense backend
#> 78  Run via pls_fit() with big.memory backend
#> 79                   Requires the pls package
#> 80              Requires the mixOmics package
#> 81       Run via pls_fit() with dense backend
#> 82  Run via pls_fit() with big.memory backend
#> 83                   Requires the pls package
#> 84              Requires the mixOmics package
#> 85       Run via pls_fit() with dense backend
#> 86  Run via pls_fit() with big.memory backend
#> 87                   Requires the pls package
#> 88              Requires the mixOmics package
#> 89       Run via pls_fit() with dense backend
#> 90  Run via pls_fit() with big.memory backend
#> 91                   Requires the pls package
#> 92              Requires the mixOmics package
#> 93       Run via pls_fit() with dense backend
#> 94  Run via pls_fit() with big.memory backend
#> 95                   Requires the pls package
#> 96              Requires the mixOmics package
#> 97       Run via pls_fit() with dense backend
#> 98  Run via pls_fit() with big.memory backend
#> 99                   Requires the pls package
#> 100             Requires the mixOmics package
#> 101      Run via pls_fit() with dense backend
#> 102 Run via pls_fit() with big.memory backend
#> 103                  Requires the pls package
#> 104             Requires the mixOmics package
#> 105      Run via pls_fit() with dense backend
#> 106 Run via pls_fit() with big.memory backend
#> 107                  Requires the pls package
#> 108             Requires the mixOmics package
#> 109      Run via pls_fit() with dense backend
#> 110 Run via pls_fit() with big.memory backend
#> 111                  Requires the pls package
#> 112             Requires the mixOmics package
#> 113      Run via pls_fit() with dense backend
#> 114 Run via pls_fit() with big.memory backend
#> 115                  Requires the pls package
#> 116             Requires the mixOmics package
#> 117      Run via pls_fit() with dense backend
#> 118 Run via pls_fit() with big.memory backend
#> 119                  Requires the pls package
#> 120             Requires the mixOmics package
#> 121      Run via pls_fit() with dense backend
#> 122 Run via pls_fit() with big.memory backend
#> 123                  Requires the pls package
#> 124             Requires the mixOmics package
#> 125      Run via pls_fit() with dense backend
#> 126 Run via pls_fit() with big.memory backend
#> 127                  Requires the pls package
#> 128             Requires the mixOmics package
#> 129      Run via pls_fit() with dense backend
#> 130 Run via pls_fit() with big.memory backend
#> 131                  Requires the pls package
#> 132             Requires the mixOmics package
#> 133      Run via pls_fit() with dense backend
#> 134 Run via pls_fit() with big.memory backend
#> 135                  Requires the pls package
#> 136             Requires the mixOmics package
#> 137      Run via pls_fit() with dense backend
#> 138 Run via pls_fit() with big.memory backend
#> 139                  Requires the pls package
#> 140             Requires the mixOmics package
#> 141      Run via pls_fit() with dense backend
#> 142 Run via pls_fit() with big.memory backend
#> 143                  Requires the pls package
#> 144             Requires the mixOmics package
#> 193      Run via pls_fit() with dense backend
#> 194 Run via pls_fit() with big.memory backend
#> 195                  Requires the pls package
#> 196             Requires the mixOmics package
#> 197      Run via pls_fit() with dense backend
#> 198 Run via pls_fit() with big.memory backend
#> 199                  Requires the pls package
#> 200             Requires the mixOmics package
#> 201      Run via pls_fit() with dense backend
#> 202 Run via pls_fit() with big.memory backend
#> 203                  Requires the pls package
#> 204             Requires the mixOmics package
#> 205      Run via pls_fit() with dense backend
#> 206 Run via pls_fit() with big.memory backend
#> 207                  Requires the pls package
#> 208             Requires the mixOmics package
#> 209      Run via pls_fit() with dense backend
#> 210 Run via pls_fit() with big.memory backend
#> 211                  Requires the pls package
#> 212             Requires the mixOmics package
#> 213      Run via pls_fit() with dense backend
#> 214 Run via pls_fit() with big.memory backend
#> 215                  Requires the pls package
#> 216             Requires the mixOmics package
#> 217      Run via pls_fit() with dense backend
#> 218 Run via pls_fit() with big.memory backend
#> 219                  Requires the pls package
#> 220             Requires the mixOmics package
#> 221      Run via pls_fit() with dense backend
#> 222 Run via pls_fit() with big.memory backend
#> 223                  Requires the pls package
#> 224             Requires the mixOmics package
#> 225      Run via pls_fit() with dense backend
#> 226 Run via pls_fit() with big.memory backend
#> 227                  Requires the pls package
#> 228             Requires the mixOmics package
#> 229      Run via pls_fit() with dense backend
#> 230 Run via pls_fit() with big.memory backend
#> 231                  Requires the pls package
#> 232             Requires the mixOmics package
#> 233      Run via pls_fit() with dense backend
#> 234 Run via pls_fit() with big.memory backend
#> 235                  Requires the pls package
#> 236             Requires the mixOmics package
#> 237      Run via pls_fit() with dense backend
#> 238 Run via pls_fit() with big.memory backend
#> 239                  Requires the pls package
#> 240             Requires the mixOmics package
#> 241      Run via pls_fit() with dense backend
#> 242 Run via pls_fit() with big.memory backend
#> 243                  Requires the pls package
#> 244             Requires the mixOmics package
#> 245      Run via pls_fit() with dense backend
#> 246 Run via pls_fit() with big.memory backend
#> 247                  Requires the pls package
#> 248             Requires the mixOmics package
#> 249      Run via pls_fit() with dense backend
#> 250 Run via pls_fit() with big.memory backend
#> 251                  Requires the pls package
#> 252             Requires the mixOmics package
#> 253      Run via pls_fit() with dense backend
#> 254 Run via pls_fit() with big.memory backend
#> 255                  Requires the pls package
#> 256             Requires the mixOmics package
#> 257      Run via pls_fit() with dense backend
#> 258 Run via pls_fit() with big.memory backend
#> 259                  Requires the pls package
#> 260             Requires the mixOmics package
#> 261      Run via pls_fit() with dense backend
#> 262 Run via pls_fit() with big.memory backend
#> 263                  Requires the pls package
#> 264             Requires the mixOmics package
#> 265      Run via pls_fit() with dense backend
#> 266 Run via pls_fit() with big.memory backend
#> 267                  Requires the pls package
#> 268             Requires the mixOmics package
#> 269      Run via pls_fit() with dense backend
#> 270 Run via pls_fit() with big.memory backend
#> 271                  Requires the pls package
#> 272             Requires the mixOmics package
#> 273      Run via pls_fit() with dense backend
#> 274 Run via pls_fit() with big.memory backend
#> 275                  Requires the pls package
#> 276             Requires the mixOmics package
#> 277      Run via pls_fit() with dense backend
#> 278 Run via pls_fit() with big.memory backend
#> 279                  Requires the pls package
#> 280             Requires the mixOmics package
#> 281      Run via pls_fit() with dense backend
#> 282 Run via pls_fit() with big.memory backend
#> 283                  Requires the pls package
#> 284             Requires the mixOmics package
#> 285      Run via pls_fit() with dense backend
#> 286 Run via pls_fit() with big.memory backend
#> 287                  Requires the pls package
#> 288             Requires the mixOmics package
#> 305      Run via pls_fit() with dense backend
#> 306 Run via pls_fit() with big.memory backend
#> 307                  Requires the pls package
#> 308             Requires the mixOmics package
#> 309      Run via pls_fit() with dense backend
#> 310 Run via pls_fit() with big.memory backend
#> 311                  Requires the pls package
#> 312             Requires the mixOmics package
#> 313      Run via pls_fit() with dense backend
#> 314 Run via pls_fit() with big.memory backend
#> 315                  Requires the pls package
#> 316             Requires the mixOmics package
#> 317      Run via pls_fit() with dense backend
#> 318 Run via pls_fit() with big.memory backend
#> 319                  Requires the pls package
#> 320             Requires the mixOmics package
#> 321      Run via pls_fit() with dense backend
#> 322 Run via pls_fit() with big.memory backend
#> 323                  Requires the pls package
#> 324             Requires the mixOmics package
#> 325      Run via pls_fit() with dense backend
#> 326 Run via pls_fit() with big.memory backend
#> 327                  Requires the pls package
#> 328             Requires the mixOmics package
#> 329      Run via pls_fit() with dense backend
#> 330 Run via pls_fit() with big.memory backend
#> 331                  Requires the pls package
#> 332             Requires the mixOmics package
#> 333      Run via pls_fit() with dense backend
#> 334 Run via pls_fit() with big.memory backend
#> 335                  Requires the pls package
#> 336             Requires the mixOmics package
#> 353      Run via pls_fit() with dense backend
#> 354 Run via pls_fit() with big.memory backend
#> 355                  Requires the pls package
#> 356             Requires the mixOmics package
#> 357      Run via pls_fit() with dense backend
#> 358 Run via pls_fit() with big.memory backend
#> 359                  Requires the pls package
#> 360             Requires the mixOmics package
#> 361      Run via pls_fit() with dense backend
#> 362 Run via pls_fit() with big.memory backend
#> 363                  Requires the pls package
#> 364             Requires the mixOmics package
#> 365      Run via pls_fit() with dense backend
#> 366 Run via pls_fit() with big.memory backend
#> 367                  Requires the pls package
#> 368             Requires the mixOmics package
#> 369      Run via pls_fit() with dense backend
#> 370 Run via pls_fit() with big.memory backend
#> 371                  Requires the pls package
#> 372             Requires the mixOmics package
#> 373      Run via pls_fit() with dense backend
#> 374 Run via pls_fit() with big.memory backend
#> 375                  Requires the pls package
#> 376             Requires the mixOmics package
#> 377      Run via pls_fit() with dense backend
#> 378 Run via pls_fit() with big.memory backend
#> 379                  Requires the pls package
#> 380             Requires the mixOmics package
#> 381      Run via pls_fit() with dense backend
#> 382 Run via pls_fit() with big.memory backend
#> 383                  Requires the pls package
#> 384             Requires the mixOmics package
```

The stored table mirrors the structure of the PLS1 benchmark and was
produced with the script in `inst/scripts/external_pls_benchmarks.R`.

## Key messages

- Dense SIMPLS remains the fastest option for well-sized dense matrices.
- Streaming NIPALS offers robustness when responses are numerous or when
  the predictor matrix is file-backed.
- External comparisons help position bigPLSR relative to established
  alternatives without adding heavyweight dependencies to the vignette.
