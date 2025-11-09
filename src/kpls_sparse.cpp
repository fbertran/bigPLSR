#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
using namespace Rcpp;

[[maybe_unused]] static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) stop(std::string(nm) + " must be a double big.matrix");
}


// ---------- Sparse KPLS (dense scaffold) -----------------------------------
// [[Rcpp::export]]
SEXP cpp_sparse_kpls_dense(const arma::mat& X,
                           const arma::mat& Y,
                           int ncomp, double tol){
  const arma::uword n = X.n_rows, p = X.n_cols, m = Y.n_cols;
  int used = std::max(0,std::min<int>(ncomp, std::min<int>(n,p)));
  arma::rowvec xm = arma::mean(X,0), ym = arma::mean(Y,0);
  arma::mat beta(p, m, arma::fill::zeros);
  arma::mat W(p, used, arma::fill::zeros), P(p, used, arma::fill::zeros);
  arma::mat Q(m, used, arma::fill::zeros), T(n, used, arma::fill::zeros);
  return List::create(
    _["coefficients"] = beta,
    _["intercept"]    = as<NumericVector>(wrap(ym - xm * beta)),
    _["x_weights"]    = W,
    _["x_loadings"]   = P,
    _["y_loadings"]   = Q,
    _["scores"]       = T,
    _["x_means"]      = as<NumericVector>(wrap(xm)),
    _["y_means"]      = as<NumericVector>(wrap(ym)),
    _["ncomp"]        = used
  );
}
