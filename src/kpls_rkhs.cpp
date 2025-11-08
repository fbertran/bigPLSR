#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
using namespace Rcpp;

static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) stop(std::string(nm) + " must be a double big.matrix");
}

// ---------- RKHS / KPLS (dense) --------------------------------------------
// [[Rcpp::export]]
SEXP cpp_kpls_rkhs_dense(const arma::mat& X,
                         const arma::mat& Y,
                         int ncomp, double tol,
                         std::string kernel, double gamma, int degree, double coef0,
                         std::string approx, int approx_rank,
                         bool return_scores){
  const arma::uword n = X.n_rows, p = X.n_cols, m = Y.n_cols;
  arma::rowvec xm = arma::mean(X,0), ym = arma::mean(Y,0);
  int used = std::max(0,std::min<int>(ncomp, std::min<int>(n, p)));
  arma::mat scores; if (return_scores && used>0){ scores.zeros(n, used); }
  arma::mat dual_alpha(n, m, arma::fill::zeros);
  return List::create(
    _["coefficients"] = R_NilValue,
    _["dual_coef"] = dual_alpha,
    _["intercept"] = as<NumericVector>(wrap(ym)),
    _["x_weights"] = R_NilValue,
    _["x_loadings"]= R_NilValue,
    _["y_loadings"]= R_NilValue,
    _["scores"]     = scores.n_elem ? wrap(scores) : R_NilValue,
    _["x_means"]    = as<NumericVector>(wrap(xm)),
    _["y_means"]    = as<NumericVector>(wrap(ym)),
    _["ncomp"]      = used
  );
}

// ---------- RKHS / KPLS (bigmem, block Gram) -------------------------------
// [[Rcpp::export]]
SEXP cpp_kpls_rkhs_bigmem(SEXP X_ptr, SEXP Y_ptr,
                          int ncomp, int chunk, double tol,
                          std::string kernel, double gamma, int degree, double coef0,
                          std::string approx, int approx_rank,
                          bool return_scores){
  XPtr<BigMatrix> xp(X_ptr); XPtr<BigMatrix> yp(Y_ptr);
  ensure_double_bigmatrix(*xp,"X"); ensure_double_bigmatrix(*yp,"Y");
  const arma::uword n = xp->nrow(), m = yp->ncol();
  int used = std::max(0,std::min<int>(ncomp, (int)n));
  arma::rowvec ym(m, arma::fill::zeros);
  arma::mat scores; if (return_scores && used>0){ scores.zeros(n, used); }
  arma::mat dual_alpha(n, m, arma::fill::zeros);
  return List::create(
    _["coefficients"] = R_NilValue,
    _["dual_coef"] = dual_alpha,
    _["intercept"] = as<NumericVector>(wrap(ym)),
    _["x_weights"] = R_NilValue,
    _["x_loadings"]= R_NilValue,
    _["y_loadings"]= R_NilValue,
    _["scores"]     = scores.n_elem ? wrap(scores) : R_NilValue,
    _["x_means"]    = R_NilValue,
    _["y_means"]    = as<NumericVector>(wrap(ym)),
    _["ncomp"]      = used,
    _["chunk_size"] = chunk
  );
}
