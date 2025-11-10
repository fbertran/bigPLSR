#' @export
kf_pls_state_new <- function(p, m, ncomp, lambda = 0.99, q_proc = 0, r_meas = 0) {
  .Call(`_bigPLSR_cpp_kf_pls_state_new`, as.integer(p), as.integer(m), as.integer(ncomp),
        as.numeric(lambda), as.numeric(q_proc), as.numeric(r_meas))
}

#' @export
kf_pls_state_update <- function(state, X_chunk, Y_chunk) {
  X_chunk <- as.matrix(X_chunk)
  .Call(`_bigPLSR_cpp_kf_pls_state_update`, state, X_chunk, Y_chunk)
  invisible(state)
}

#' Stateful PLS Kalman Filter
#' 
#' @export
#' @examples
#' n <- 1000; p <- 30; m <- 2; A <- 3
#' X <- matrix(rnorm(n*p), n, p)
#' Y <- X[,1:2] %*% matrix(c(0.7, -0.3, 0.2, 0.9), 2, m) + matrix(rnorm(n*m, sd=0.2), n, m)
#' 
#' state <- kf_pls_state_new(p, m, A, lambda = 0.99, q_proc = 1e-6)
#' 
#' # stream in mini-batches
#' bs <- 128
#' for (i in seq(1, n, by = bs)) {
#'   idx <- i:min(i+bs-1, n)
#'   kf_pls_state_update(state, X[idx, , drop=FALSE], Y[idx, , drop=FALSE])
#' }
#' 
#' fit <- kf_pls_state_fit(state)  # returns a big_plsr-compatible list
#' # predict via your existing predict.big_plsr (linear case)
#' Yhat <- cbind(1, scale(X, center = fit$x_means, scale = FALSE)) %*%
#'   rbind(fit$intercept, fit$coefficients)
#' 
kf_pls_state_fit <- function(state, tol = 1e-8) {
  fit <- .Call(`_bigPLSR_cpp_kf_pls_state_fit`, state, as.numeric(tol))
  class(fit) <- unique(c("big_plsr", class(fit)))
  fit
}
