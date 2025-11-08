#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
using namespace Rcpp;

static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) stop(std::string(nm) + " must be a double big.matrix");
}


// ---------- Double RKHS (dense scaffold) -----------------------------------
// [[Rcpp::export]]
SEXP cpp_rkhs_xy_dense(const arma::mat& X,
                       const arma::mat& Y,
                       int ncomp, double tol,
                       std::string kernel, double gamma, int degree, double coef0){
  const arma::uword n = X.n_rows, p = X.n_cols, m = Y.n_cols;
  arma::rowvec xm = arma::mean(X,0), ym = arma::mean(Y,0);
  int used = std::max(0,std::min<int>(ncomp, std::min<int>(n,p)));
  arma::mat dual_alpha(n, m, arma::fill::zeros);
  arma::mat T; if (used>0) T.zeros(n, used);
  return List::create(
    _["coefficients"] = R_NilValue,
    _["dual_coef"]    = dual_alpha,
    _["intercept"]    = as<NumericVector>(wrap(ym)),
    _["x_weights"]    = R_NilValue,
    _["x_loadings"]   = R_NilValue,
    _["y_loadings"]   = R_NilValue,
    _["scores"]       = T.n_elem ? wrap(T) : R_NilValue,
    _["x_means"]      = as<NumericVector>(wrap(xm)),
    _["y_means"]      = as<NumericVector>(wrap(ym)),
    _["ncomp"]        = used
  );
}
