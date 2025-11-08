#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;

// ---------- helpers ---------------------------------------------------------

static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) {
    stop(std::string(nm) + " must be a double big.matrix");
  }
}

// pull a row subset (1-based indices from R) from a BigMatrix into an R matrix
static inline NumericMatrix rows_from_bigmatrix(BigMatrix& BM,
                                                const IntegerVector& idx_1based){
  const std::size_t ni = idx_1based.size();
  const std::size_t p  = BM.ncol();
  MatrixAccessor<double> acc(BM);
  
  NumericMatrix out(ni, p);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = acc[j];
    for (std::size_t r = 0; r < ni; ++r) {
      int i0 = idx_1based[r] - 1; // 1->0 based
      if (i0 < 0 || i0 >= (int)BM.nrow()) stop("row index out of range");
      out(r, j) = col[i0];
    }
  }
  return out;
}

// ---------- 3) Kernel logistic PLS (skeleton) -------------------------------
//
// Minimal stub: validates shapes and returns zeroed parameters with
// consistent dimensions so R wrappers and tests can link. Replace with
// IRLS + KPLS latent construction later.
//
// [[Rcpp::export]]
SEXP cpp_klogit_pls_fit(const arma::mat& X,        // n x p (optional: may switch to K)
                        const arma::vec& y,        // length n, 0/1
                        int ncomp,
                        double tol = 1e-6,
                        int maxit = 25) {
  if (X.n_rows == 0 || X.n_cols == 0) stop("X must be non-empty");
  if (y.n_elem != X.n_rows) stop("length(y) must equal nrow(X)");
  if (ncomp < 0) ncomp = 0;
  
  const arma::uword n = X.n_rows;
  const arma::uword p = X.n_cols;
  
  arma::rowvec x_means = arma::mean(X, 0);
  arma::mat beta(p, 1, arma::fill::zeros);
  arma::vec intercept(1, arma::fill::zeros);
  arma::mat scores;
  if (ncomp > 0) { scores.set_size(n, (arma::uword)ncomp); scores.zeros(); }
  
  return List::create(
    _["coefficients"] = beta,                 // p x 1 (zeros)
    _["intercept"]    = intercept,            // length 1
    _["x_weights"]    = R_NilValue,
    _["x_loadings"]   = R_NilValue,
    _["y_loadings"]   = R_NilValue,
    _["scores"]       = scores.n_elem ? wrap(scores) : R_NilValue,
    _["x_means"]      = as<NumericVector>(wrap(x_means)),
    _["y_means"]      = NumericVector::create(mean(y)),
    _["ncomp"]        = std::max(0, ncomp),
    _["converged"]    = false,
    _["iter"]         = 0
  );
}

// ---------- Kernel logistic PLS (dense) ------------------------------------
// [[Rcpp::export]]
SEXP cpp_klogit_pls_dense(const arma::mat& X,
                          const arma::vec& y,
                          int ncomp, double tol,
                          std::string kernel, double gamma, int degree, double coef0,
                          arma::vec class_weights){
  const arma::uword n = X.n_rows, p = X.n_cols;
  if (y.n_elem != n) stop("y length must match nrow(X)");
  arma::rowvec xm = arma::mean(X,0);
  arma::mat beta_latent(ncomp>0?ncomp:1,1,arma::fill::zeros);
  arma::vec intercept(1,arma::fill::zeros);
  arma::mat scores; if (ncomp>0){ scores.zeros(n, ncomp); }
  return List::create(
    _["coefficients"] = beta_latent,
    _["intercept"]    = intercept,
    _["x_weights"]    = R_NilValue,
    _["x_loadings"]   = R_NilValue,
    _["y_loadings"]   = R_NilValue,
    _["scores"]       = scores.n_elem ? wrap(scores) : R_NilValue,
    _["x_means"]      = as<NumericVector>(wrap(xm)),
    _["y_means"]      = NumericVector::create(arma::mean(y)),
    _["ncomp"]        = std::max(0,ncomp),
    _["converged"]    = false,
    _["iter"]         = 0
  );
}

// ---------- Kernel logistic PLS (bigmem) -----------------------------------
// [[Rcpp::export]]
SEXP cpp_klogit_pls_bigmem(SEXP X_ptr,
                           arma::vec y,
                           int ncomp, int chunk, double tol,
                           std::string kernel, double gamma, int degree, double coef0,
                           arma::vec class_weights){
  XPtr<BigMatrix> xp(X_ptr); ensure_double_bigmatrix(*xp,"X");
  const arma::uword n = xp->nrow();
  if (y.n_elem != n) stop("y length must match nrow(X)");
  arma::mat scores; if (ncomp>0){ scores.zeros(n, ncomp); }
  return List::create(
    _["coefficients"] = arma::mat(ncomp>0?ncomp:1,1,arma::fill::zeros),
    _["intercept"]    = arma::vec(1,arma::fill::zeros),
    _["scores"]       = scores.n_elem ? wrap(scores) : R_NilValue,
    _["ncomp"]        = std::max(0,ncomp),
    _["chunk_size"]   = chunk,
    _["converged"]    = false
  );
}