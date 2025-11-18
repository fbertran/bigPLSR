#' KF-PLS streaming state (constructor)
#'
#' Create a persistent Kalman–filter PLS (KF-PLS) state that accumulates
#' cross-products from streaming mini-batches and later produces a
#' `big_plsr`-compatible fit via [`kf_pls_state_fit()`].
#'
#' The state maintains exponentially weighted cross-moments
#' \eqn{C_{xx}} and \eqn{C_{xy}} with forgetting factor `lambda`.
#' When `lambda >= 0.999999` and `q_proc == 0`, the backend switches to an
#' *exact* accumulation mode that matches concatenating all chunks (no decay).
#'
#' @param p Integer, number of predictors (columns of `X`).
#' @param m Integer, number of responses (columns of `Y`).
#' @param ncomp Integer, number of latent components to extract at fit time.
#' @param lambda Numeric in (0,1], forgetting factor (closer to 1 = slower decay).
#' @param q_proc Non-negative numeric, process-noise magnitude (adds a ridge to
#'   \eqn{C_{xx}} each update; useful for stabilizing ill-conditioned problems).
#' @param r_meas Reserved measurement-noise parameter (not used by the minimal
#'   API yet; kept for forward compatibility).
#'
#' @return An external pointer to an internal KF-PLS state (opaque object) that
#'   you pass to [`kf_pls_state_update()`] and then to
#'   [`kf_pls_state_fit()`] to produce model coefficients.
#'
#' @seealso [kf_pls_state_update()], [kf_pls_state_fit()], [pls_fit()]
#'   (use `algorithm = "kf_pls"` for the one-shot dense path).
#' @examples
#' \dontrun{
#' set.seed(1)
#' n <- 1000; p <- 50; m <- 2
#' X1 <- matrix(rnorm(n/2 * p), n/2, p)
#' X2 <- matrix(rnorm(n/2 * p), n/2, p)
#' B  <- matrix(rnorm(p*m), p, m)
#' Y1 <- scale(X1, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n/2*m), n/2, m)
#' Y2 <- scale(X2, TRUE, FALSE) %*% B + 0.05*matrix(rnorm(n/2*m), n/2, m)
#'
#' st <- kf_pls_state_new(p, m, ncomp = 4, lambda = 0.99, q_proc = 1e-6)
#' kf_pls_state_update(st, X1, Y1)
#' kf_pls_state_update(st, X2, Y2)
#' fit <- kf_pls_state_fit(st)          # returns a big_plsr-compatible list
#' preds <- predict(.finalize_pls_fit(fit, "kf_pls"), rbind(X1, X2))
#' }
#' @export
kf_pls_state_new <- function(p, m, ncomp, lambda = 0.99, q_proc = 0, r_meas = 0) {
  .Call(`_bigPLSR_cpp_kf_pls_state_new`, as.integer(p), as.integer(m), as.integer(ncomp),
        as.numeric(lambda), as.numeric(q_proc), as.numeric(r_meas))
}


#' Update a KF-PLS streaming state with a mini-batch
#'
#' Feed one chunk (`X_chunk`, `Y_chunk`) to an existing KF-PLS state created by
#' [`kf_pls_state_new()`]. The function updates exponentially weighted means and
#' cross-products (or exact sufficient statistics when in exact mode).
#'
#' @param state External pointer produced by [kf_pls_state_new()].
#' @param X_chunk Numeric matrix with the same number of columns `p` used to
#'   create the state.
#' @param Y_chunk Numeric matrix with `m` columns (or a numeric vector if
#'   `m == 1`). Must have the same number of rows as `X_chunk`.
#'
#' @return Invisibly returns `state`, updated in place.
#'
#' @details
#' Call this repeatedly for each incoming batch. When you want model
#' coefficients (weights/loadings/intercepts), call
#' [`kf_pls_state_fit()`], which solves SIMPLS on the accumulated
#' cross-moments without re-materializing all past data.
#'
#' @seealso [kf_pls_state_new()], [kf_pls_state_fit()]
#' @export
kf_pls_state_update <- function(state, X_chunk, Y_chunk) {
  X_chunk <- as.matrix(X_chunk)
  .Call(`_bigPLSR_cpp_kf_pls_state_update`, state, X_chunk, Y_chunk)
  invisible(state)
}

#' Finalize a KF-PLS state into a fitted model
#'
#' Converts the accumulated KF-PLS state into a SIMPLS-equivalent fitted
#' model (using the current sufficient statistics). The result is compatible
#' with [predict.big_plsr()].
#'
#' @param state External pointer created by [kf_pls_state_new()].
#' @param tol Numeric tolerance for the inner SIMPLS step.
#'
#' @return A list with PLS factors and coefficients, classed as `big_plsr`.
#' @export
#' @examples
#' n <- 200; p <- 30; m <- 2; A <- 3
#' X <- matrix(rnorm(n*p), n, p)
#' Y <- X[,1:2] %*% matrix(c(0.7, -0.3, 0.2, 0.9), 2, m) + matrix(rnorm(n*m, sd=0.2), n, m)
#' 
#' state <- kf_pls_state_new(p, m, A, lambda = 0.99, q_proc = 1e-6)
#' 
#' # stream in mini-batches
#' bs <- 64
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
  if (!is.null(fit$intercept)) fit$intercept <- as.numeric(fit$intercept)
  if (!is.null(fit$coefficients)) fit$coefficients <- as.matrix(fit$coefficients)
  class(fit) <- unique(c("big_plsr", class(fit)))
  fit
}
