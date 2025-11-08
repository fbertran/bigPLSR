#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
using namespace Rcpp;

static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) stop(std::string(nm) + " must be a double big.matrix");
}


// ---------- Kalman-filter PLS (bigmem stream scaffold) ---------------------
// [[Rcpp::export]]
SEXP cpp_kf_pls_stream(SEXP X_ptr, SEXP Y_ptr,
                       int ncomp, int chunk, double tol){
  XPtr<BigMatrix> xp(X_ptr); XPtr<BigMatrix> yp(Y_ptr);
  ensure_double_bigmatrix(*xp,"X"); ensure_double_bigmatrix(*yp,"Y");
  const arma::uword n = xp->nrow(), m = yp->ncol();
  int used = std::max(0,ncomp);
  arma::mat T; if (used>0) T.zeros(n, used);
  return List::create(
    _["coefficients"] = R_NilValue,
    _["intercept"]    = arma::rowvec(m, arma::fill::zeros),
    _["x_weights"]    = R_NilValue,
    _["x_loadings"]   = R_NilValue,
    _["y_loadings"]   = R_NilValue,
    _["scores"]       = T.n_elem ? wrap(T) : R_NilValue,
    _["ncomp"]        = used,
    _["chunk_size"]   = chunk,
    _["state"]        = List::create(_["initialized"]=false)
  );
}
