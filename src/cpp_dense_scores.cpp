#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "cblas_compat.h"

// T = (X - 1 * mu_x) %*% W   computed as T = X W - 1 (mu_x W)
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_dense_scores(Rcpp::NumericMatrix X_,
                                     Rcpp::NumericVector mu_x_,
                                     Rcpp::NumericMatrix W_)
{
  arma::mat X (X_.begin(),  X_.nrow(),  X_.ncol(),  false, true);
  arma::rowvec mu_x(mu_x_.begin(), mu_x_.size(), false, true);
  arma::mat W (W_.begin(),  W_.nrow(),  W_.ncol(),  false, true);
  const int n = (int)X.n_rows;
  const int k = (int)W.n_cols;
  
  arma::mat T(n, k, arma::fill::zeros);
  
#if BIGPLSR_HAVE_CBLAS
  // T = X W
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
              n, k, (int)X.n_cols,
              1.0, X.memptr(), n,
              W.memptr(), (int)W.n_rows,
              0.0, T.memptr(), n);
#else
  T = X * W;
#endif
  
  // shift = mu_x W   (1 x k), then subtract from each column
  arma::rowvec shift = mu_x * W;
  T.each_row() -= shift;
  return Rcpp::NumericMatrix(Rcpp::wrap(T));
}
