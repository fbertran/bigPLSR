#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;
using arma::uword;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

// [[Rcpp::plugins(cpp17)]]

// Small helper: enforce double big.matrix
static inline void ensure_double_matrix(const BigMatrix& mat, const char* name) {
  if (mat.matrix_type() != 8) {
    Rcpp::stop(std::string(name) + " must be a double precision big.matrix");
  }
}

// ---- Dense KF-PLS (EWMA of cross-products) ---------------------------------
//
// We maintain exponentially-weighted cross-products:
//   Cxx <- lambda * Cxx + XtX_batch + q_proc * I
//   Cxy <- lambda * Cxy + XtY_batch
// Then call the existing SIMPLS-from-cross routine in C++.
//
// Returns the usual PLS fit list (coef, intercept, W, P, Q, scores=NULL here).
//
// [[Rcpp::export]]
Rcpp::List cpp_kf_pls_dense(Rcpp::NumericMatrix X_,
                            Rcpp::RObject Y_,
                            int ncomp,
                            double tol,
                            double lambda,
                            double q_proc) {
  if (ncomp <= 0) Rcpp::stop("ncomp must be positive");
  arma::mat X(X_.begin(), X_.nrow(), X_.ncol(), false, true);
  arma::mat Y;
  if (Rf_isMatrix(Y_)) {
    Rcpp::NumericMatrix Ym(Y_);
    if ((int)Ym.nrow() != X_.nrow()) Rcpp::stop("X and Y row mismatch");
    Y = arma::mat(Ym.begin(), Ym.nrow(), Ym.ncol(), false, true);
  } else {
    Rcpp::NumericVector yv(Y_);
    if ((int)yv.size() != X_.nrow()) Rcpp::stop("X and y length mismatch");
    Y = arma::mat(yv.begin(), X_.nrow(), 1, false, true);
  }
  
  // Center means for intercept consistency
  arma::rowvec x_means = arma::mean(X, 0);
  arma::rowvec y_means = arma::mean(Y, 0);
  arma::mat Xc = X.each_row() - x_means;
  arma::mat Yc = Y.each_row() - y_means;
  
  arma::mat Cxx = arma::zeros<arma::mat>(Xc.n_cols, Xc.n_cols);
  arma::mat Cxy = arma::zeros<arma::mat>(Xc.n_cols, Yc.n_cols);
  
  // Single “batch” pass (dense); EWMA still meaningful for multi-call updates
  Cxx = lambda * Cxx + Xc.t() * Xc;
  Cxy = lambda * Cxy + Xc.t() * Yc;
  if (q_proc > 0) Cxx += q_proc * arma::eye<arma::mat>(Cxx.n_rows, Cxx.n_cols);
  
  // Call the SIMPLS-from-cross exported symbol
  // NOTE:
  //   - From C++ we call the **R wrapper** `cpp_simpls_from_cross`
  //     living in the bigPLSR namespace, not the C symbol with leading "_".
  //   - Make argument types explicit to avoid any ambiguity.
  
  // Convert arma to R objects explicitly
  Rcpp::NumericMatrix XtX(Cxx.n_rows, Cxx.n_cols);
  Rcpp::NumericMatrix XtY(Cxy.n_rows, Cxy.n_cols);
  std::copy(Cxx.begin(), Cxx.end(), XtX.begin());
  std::copy(Cxy.begin(), Cxy.end(), XtY.begin());
  
  Rcpp::NumericVector xmean(x_means.begin(), x_means.end());
  Rcpp::NumericVector ymean(y_means.begin(), y_means.end());
  
  // Lookup the R function bigPLSR::cpp_simpls_from_cross
  Rcpp::Environment ns = Rcpp::Environment::namespace_env("bigPLSR");
  Rcpp::Function cpp_simpls_from_cross = ns["cpp_simpls_from_cross"];
  
  // Call it: cpp_simpls_from_cross(XtX, XtY, x_means, y_means, ncomp, tol)
  Rcpp::List fit = cpp_simpls_from_cross(
    XtX,
    XtY,
    xmean,
    ymean,
    Rcpp::wrap(ncomp),
    Rcpp::wrap(tol)
  );
  // Scores are handled at the R layer if requested (dense path).
  return fit;
}

// ---- Big-memory KF-PLS (streamed EWMA of cross-products) -------------------
//
// Stream blocks of rows, update EWMA cross-products, then run SIMPLS-from-cross.
//
// [[Rcpp::export]]
Rcpp::List cpp_kf_pls_bigmem(SEXP X_ptr,
                             SEXP Y_ptr,
                             int ncomp,
                             int chunk_rows,
                             double tol,
                             double lambda,
                             double q_proc) {
  if (TYPEOF(X_ptr) != EXTPTRSXP || TYPEOF(Y_ptr) != EXTPTRSXP)
    Rcpp::stop("X_ptr and Y_ptr must be big.matrix external pointers");
  Rcpp::XPtr<BigMatrix> Xbm(X_ptr);
  Rcpp::XPtr<BigMatrix> Ybm(Y_ptr);
  ensure_double_matrix(*Xbm, "X");
  ensure_double_matrix(*Ybm, "Y");
  if ((uword)Xbm->nrow() != (uword)Ybm->nrow()) Rcpp::stop("X and Y row mismatch");
  if (ncomp <= 0) Rcpp::stop("ncomp must be positive");
  if (chunk_rows <= 0) Rcpp::stop("chunk_rows must be > 0");
  
  const uword n = Xbm->nrow();
  const uword p = Xbm->ncol();
  const uword m = Ybm->ncol();
  
  MatrixAccessor<double> Xacc(*Xbm);
  MatrixAccessor<double> Yacc(*Ybm);
  
  // Compute means in one streamed pass
  arma::rowvec x_means(p, arma::fill::zeros);
  arma::rowvec y_means(m, arma::fill::zeros);
  {
    arma::rowvec sumx(p, arma::fill::zeros);
    arma::rowvec sumy(m, arma::fill::zeros);
    for (uword i0 = 0; i0 < n; i0 += (uword)chunk_rows) {
      uword i1 = std::min<uword>(n, i0 + (uword)chunk_rows);
      uword nr = i1 - i0;
      arma::mat Xblk(nr, p);
      arma::mat Yblk(nr, m);
      for (uword j = 0; j < p; ++j) for (uword i = 0; i < nr; ++i) Xblk(i,j) = Xacc[j][i0 + i];
      for (uword j = 0; j < m; ++j) for (uword i = 0; i < nr; ++i) Yblk(i,j) = Yacc[j][i0 + i];
      sumx += arma::sum(Xblk, 0);
      sumy += arma::sum(Yblk, 0);
    }
    x_means = sumx / double(n);
    y_means = sumy / double(n);
  }
  
  arma::mat Cxx(p, p, arma::fill::zeros);
  arma::mat Cxy(p, m, arma::fill::zeros);
  
  // Stream EWMA cross-products on centered blocks
  for (uword i0 = 0; i0 < n; i0 += (uword)chunk_rows) {
    uword i1 = std::min<uword>(n, i0 + (uword)chunk_rows);
    uword nr = i1 - i0;
    arma::mat Xblk(nr, p);
    arma::mat Yblk(nr, m);
    for (uword j = 0; j < p; ++j) for (uword i = 0; i < nr; ++i) Xblk(i,j) = Xacc[j][i0 + i];
    for (uword j = 0; j < m; ++j) for (uword i = 0; i < nr; ++i) Yblk(i,j) = Yacc[j][i0 + i];
    Xblk.each_row() -= x_means;
    Yblk.each_row() -= y_means;
    
    Cxx = lambda * Cxx + Xblk.t() * Xblk;
    Cxy = lambda * Cxy + Xblk.t() * Yblk;
  }
  if (q_proc > 0) Cxx += q_proc * arma::eye<arma::mat>(p, p);
  
  // Call the SIMPLS-from-cross exported symbol
  // NOTE:
  //   - From C++ we call the **R wrapper** `cpp_simpls_from_cross`
  //     living in the bigPLSR namespace, not the C symbol with leading "_".
  //   - Make argument types explicit to avoid any ambiguity.
  
  // Convert arma to R objects explicitly
  Rcpp::NumericMatrix XtX(Cxx.n_rows, Cxx.n_cols);
  Rcpp::NumericMatrix XtY(Cxy.n_rows, Cxy.n_cols);
  std::copy(Cxx.begin(), Cxx.end(), XtX.begin());
  std::copy(Cxy.begin(), Cxy.end(), XtY.begin());
  
  Rcpp::NumericVector xmean(x_means.begin(), x_means.end());
  Rcpp::NumericVector ymean(y_means.begin(), y_means.end());
  
  // Lookup the R function bigPLSR::cpp_simpls_from_cross
  Rcpp::Environment ns = Rcpp::Environment::namespace_env("bigPLSR");
  Rcpp::Function cpp_simpls_from_cross = ns["cpp_simpls_from_cross"];

  // Call it: cpp_simpls_from_cross(XtX, XtY, x_means, y_means, ncomp, tol)
  Rcpp::List fit = cpp_simpls_from_cross(
    XtX,
    XtY,
    xmean,
    ymean,
    Rcpp::wrap(ncomp),
    Rcpp::wrap(tol)
  );
  return fit;
}




