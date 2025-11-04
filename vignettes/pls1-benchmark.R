## ----setup_ops, include = FALSE-----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.path = "figures/benchmarking-",
  fig.width = 7,
  fig.height = 5,
  dpi = 150,
  message = FALSE,
  warning = FALSE
)

LOCAL <- identical(Sys.getenv("LOCAL"), "TRUE")

## ----setup, message=FALSE-----------------------------------------------------
library(bigPLSR)
library(bigmemory)
library(bench)
set.seed(123)

## ----data-generation----------------------------------------------------------
n <- 1000
p <- 50
ncomp <- 5

X <- bigmemory::big.matrix(nrow = n, ncol = p, type = "double")
X[,] <- matrix(rnorm(n * p), nrow = n)

true_beta <- matrix(rnorm(p), ncol = 1)
y_vec <- as.vector(scale(X[,] %*% true_beta + rnorm(n)))

y <- bigmemory::big.matrix(nrow = n, ncol = 1, type = "double")
y[,] <- y_vec

X[1:6, 1:6]
y[1:6,]

