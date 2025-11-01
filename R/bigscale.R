#' Title
#'
#' @param formula 
#' @param data 
#' @param norm.method 
#' @param strata.size 
#' @param batch.size 
#' @param features.mean 
#' @param features.sd 
#' @param parallel.flag 
#' @param num.cores 
#' @param bigmemory.flag 
#' @param num.rows.chunk 
#' @param col.names 
#' @param type 
#'
#' @return
#'   an object of the scaler class
#'   time.indices: indices of the time variable
#'   cens.indices: indices of the censored variables 
#'   features.indices: indices of the features 
#'   time.sd: standard deviation of the time variable
#'   time.mean: mean of the time variable
#'   features.sd: standard deviation of the features
#'   features.mean: mean of the features
#'   nr: number of rows
#'   nc: number of columns
#'   col.names: columns names
#' @export
#'
#' @examples
#' 
#' 1+1
#' 
bigscale <- function (formula = Surv(time = time, status = status) ~ ., data, 
          norm.method = "standardize", strata.size = 20,
          batch.size = 1, features.mean = NULL, features.sd = NULL, 
          parallel.flag = FALSE, num.cores = NULL, bigmemory.flag = FALSE, 
          num.rows.chunk = 1e+06, col.names = NULL, type="short") 
{
  if (!bigmemory.flag) {
    if (!is.data.frame(data)) {
      big.data <- read.csv(file = data)
    }
    else {
      big.data <- data
    }
  }
  else {
    if(is.big.matrix(data)){big.data <- data} else{
      if (!is.null(col.names)) {
        big.data <- bigmemory::read.big.matrix(filename = data, sep = ",", 
                                    skip = 0, header = TRUE, 
                                    col.names = col.names, type=type)
      }
      else {
        big.data <- bigmemory::read.big.matrix(filename = data, sep = ",", 
                                    skip = 0, header = TRUE, type=type)
      }
    }
  }
  col.names <- colnames(big.data)
  num.rows.big <- nrow(big.data)
  num.cols.big <- ncol(big.data)
  if (ncol(big.data) < 3) {
    stop("data must have 3 or more columns: time, status, and at least one feature")
  }
  if (nrow(big.data) < 2) {
    stop("Sample size is too small (with size less than 2)")
  }
  all.variables <- all.vars(formula)
  time.indices <- match(all.variables[1], colnames(big.data))
  cens.indices <- match(all.variables[2], colnames(big.data))
  surv.indices <- c(time.indices, cens.indices)
  
  if (length(all.variables) == 3 & all.variables[3] == ".") {
    orig.features.indices <- setdiff(1:NCOL(big.data), surv.indices)
    features.indices <- c(time.indices,orig.features.indices)
    sub.col.names <- colnames(big.data)[features.indices]
  }
  else {
    orig.features.indices <- match(all.variables[3:length(all.variables)], 
                              colnames(big.data))
    features.indices <- c(time.indices,orig.features.indices)
    sub.col.names <- all.variables[3:length(all.variables)]
  }
  chengeStrataBatch <- (strata.size > floor(num.rows.big/batch.size)) & 
    (strata.size > 2)
  while ((strata.size > floor(num.rows.big/batch.size)) & (strata.size > 
                                                           2)) {
    if (batch.size > 1) {
      batch.size <- max(floor(batch.size/2), 1)
    }
    else {
      strata.size <- max(floor(num.rows.big/batch.size), 
                         2)
    }
  }
  if (chengeStrataBatch) {
    warning(paste0("Strata size times batch size is greater than number of observations.\n This package resizes them to strata size = ", 
                   strata.size, " and batch size = ", batch.size))
  }
  num.sub.sample <- floor(num.rows.big/num.rows.chunk)
  chunks.length <- c(0, rep(num.rows.chunk, floor(num.rows.big/num.rows.chunk)), 
                     if (num.rows.big%%num.rows.chunk != 0) {
                       num.rows.big%%num.rows.chunk
                     })
  if (is.null(features.mean) & is.null(features.sd)) {
    if (norm.method == "center") {
      n2 <- 0
      features.mean <- 0
      for (i in 1:(length(chunks.length) - 1)) {
        indices.chunk <- (sum(chunks.length[1:i]) + 1):(sum(chunks.length[1:(i + 
                                                                               1)]))
        sub.data <- big.data[indices.chunk, features.indices]
        n1 <- NROW(sub.data)
        if (NCOL(sub.data) > 1) {
          M1 <- colMeans(sub.data, na.rm = TRUE)
        }
        else {
          M1 <- mean(sub.data, na.rm = TRUE)
        }
        features.mean <- (n1 * M1 + n2 * features.mean)/(n1 + n2)
        n2 <- n1 + n2
      }
      features.sd <- rep(1, NCOL(sub.data))
    }
    else if (norm.method == "scale" || norm.method == "standardize") {
      n2 <- 0
      features.mean <- 0
      features.sd <- 0
      for (i in 1:(length(chunks.length) - 1)) {
        indices.chunk <- (sum(chunks.length[1:i]) + 1):(sum(chunks.length[1:(i + 
                                                                               1)]))
        sub.data <- big.data[indices.chunk, features.indices]
        n1 <- NROW(sub.data)
        if (NCOL(sub.data) > 1) {
          M1 <- colMeans(sub.data, na.rm = TRUE)
        }
        else {
          M1 <- mean(sub.data, na.rm = TRUE)
        }
        if (NCOL(sub.data) > 1) {
          S1 <- colMeans(sub.data^2, na.rm = TRUE) - 
            M1^2
        }
        else {
          S1 <- mean(sub.data^2, na.rm = TRUE) - M1^2
        }
        M2 <- features.mean
        S2 <- features.sd
        features.mean <- (n1 * M1 + n2 * M2)/(n1 + n2)
        features.sd <- 1/(n1 + n2) * (n1 * S1 + n2 * 
                                        S2 + (n1 * n2)/(n1 + n2) * (M1 - M2)^2)
        n2 <- n1 + n2
      }
      features.sd <- sqrt(features.sd)
    }
    else {
      features.mean <- rep(0, length(features.indices))
      features.sd <- rep(1, length(features.indices))
    }
  }
  if (sum(features.sd == 0) > 0) {
    stop(paste0("feature(s) ", colnames(big.data)[features.indices][which(features.sd == 
                                                                            0)], " is/are constant without any variability"))
  }

  out <- NULL
  out$time.indices <- time.indices
  out$cens.indices <- cens.indices
  out$features.indices <- orig.features.indices
  out$time.sd <- features.sd[1]
  out$time.mean <- features.mean[1]
  out$features.sd <- features.sd[-1]
  out$features.mean <- features.mean[-1]
  out$nr <- num.rows.big
  out$nc <- num.cols.big
  out$col.names <- col.names
  class(out) <- "scaler"
  out
}
