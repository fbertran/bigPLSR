#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

// Unified CBLAS detection (Apple vecLib or generic) with graceful fallback.
#include "cblas_compat.h"


/*
 Compute centered cross-products without materializing Xc or Yc:
 XtX = X'X - n * (mu_x' * mu_x)
 XtY = X'Y - n * (mu_x' * mu_y)
 This leverages BLAS (DGEMM/DSYRK) for the heavy multiplies and avoids the big
 sweep(X, mu) temporary.
 */

// [[Rcpp::export]]
Rcpp::List cpp_dense_cross(Rcpp::NumericMatrix X_,
                           Rcpp::NumericMatrix Y_,
                           Rcpp::Nullable<Rcpp::LogicalVector> use_syrk_opt = R_NilValue,
                           Rcpp::Nullable<Rcpp::LogicalVector> use_dgemm_opt = R_NilValue) {
  const int n = X_.nrow();
  const int p = X_.ncol();
  const int m = Y_.ncol();
  if (Y_.nrow() != n) Rcpp::stop("X and Y must have the same number of rows");
  
  // Alias R memory (no copy)
  arma::mat X(X_.begin(), n, p, /*copy_aux_mem=*/false, /*strict=*/true);
  arma::mat Y(Y_.begin(), n, m, false, true);

  // Column means via BLAS GEMV:
  //   mu_x = (X' * 1_n) / n,  mu_y = (Y' * 1_n) / n
  arma::vec ones(n, arma::fill::ones);
  arma::rowvec mu_x = (X.t() * ones).t() / double(n);
  arma::rowvec mu_y = (Y.t() * ones).t() / double(n);
  
  // BLAS-backed multiplies
  arma::mat XtX(p, p, arma::fill::zeros);
  arma::mat XtY(p, m, arma::fill::zeros);
  // Decide SYRK use (CBLAS upper-tri) if available + requested
  bool use_syrk = true;
  bool use_dgemm = true;
  if (use_syrk_opt.isNotNull()) {
    Rcpp::LogicalVector vv(use_syrk_opt.get());
    if (vv.size() > 0) use_syrk = (vv[0] == TRUE);
  }
  if (use_dgemm_opt.isNotNull()) {
    Rcpp::LogicalVector vv(use_dgemm_opt.get());
    if (vv.size() > 0) use_dgemm = (vv[0] == TRUE);
  }
#if BIGPLSR_HAVE_CBLAS
  if (use_syrk) {
    // dsyrk (ColMajor, Upper, Trans): XtX = X'X
    // n=p (rows of XtX), k=n (rows of X), lda=n, ldc=p
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                (int)p, (int)n, 1.0,
                X.memptr(), (int)n,
                0.0,
                XtX.memptr(), (int)p);
    // Mirror upper → full symmetric
    XtX = arma::symmatu(XtX);
  } else {
    // Portable path; will call BLAS if available
    XtX = X.t() * X;    
    XtX = arma::symmatu(XtX);
   }
  }
#else
  (void)use_syrk; // quiet unused
  // Portable path; will call BLAS if available
  XtX = X.t() * X;
  XtX = arma::symmatu(XtX);
#endif
  
#if BIGPLSR_HAVE_CBLAS
  if (use_dgemm) {
    // XtY = X'Y via dgemm
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                (int)p, (int)m, (int)n,
                1.0, X.memptr(), (int)n,
                Y.memptr(), (int)n,
                0.0, XtY.memptr(), (int)p);
  } else {
    // Portable path; will call BLAS if available
    XtY = X.t() * Y;
  }
#else
  (void)use_dgemm;
    // Portable path; will call BLAS if available
    XtY = X.t() * Y;
#endif

  // Rank-1 corrections: XtX -= n * (mu_x^T * mu_x), XtY -= n * (mu_x^T * mu_y)
  const double nd = static_cast<double>(n);
  XtX -= nd * (mu_x.t() * mu_x);
  XtY -= nd * (mu_x.t() * mu_y);
  
  return Rcpp::List::create(
    Rcpp::Named("XtX")     = XtX,
    Rcpp::Named("XtY")     = XtY,
    Rcpp::Named("x_means") = mu_x,
    Rcpp::Named("y_means") = mu_y
  );
}