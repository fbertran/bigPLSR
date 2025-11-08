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
static inline NumericMatrix rows_from_bigmatrix(BigMatrix& BM, const IntegerVector& idx_1based){
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

// // [[Rcpp::export]]
// SEXP cpp_kpls_from_gram(const arma::mat& K, const arma::mat& Y, int ncomp, double tol) {
//   Rcpp::stop("cpp_kpls_from_gram: not implemented in this build");
// }
// 
// // [[Rcpp::export]]
// SEXP cpp_kernel_gram_block(SEXP X_ptr,
//                            int block_rows,
//                            int block_cols,
//                            std::string kernel = "rbf",
//                            double gamma = 1.0,
//                            int degree = 3,
//                            double coef0 = 1.0) {
//   Rcpp::stop("cpp_kernel_gram_block: not implemented in this build");
// }
// 
// SEXP cpp_klogit_pls_fit(const arma::mat& T, const arma::vec& y,
//                         int maxit = 25, double lambda = 0.0) {
//   Rcpp::stop("cpp_klogit_pls_fit: not implemented in this build");
// }
// 
// SEXP cpp_sparse_kpls_fit(const arma::mat& K, const arma::mat& Y,
//                          int ncomp, double lambda, double tol) {
//   Rcpp::stop("cpp_sparse_kpls_fit: not implemented in this build");
// }
// 
// // [[Rcpp::export]]
// SEXP cpp_rkhs_xy_dense(const arma::mat& Kx, const arma::mat& Ky,
//                        int ncomp, double ridge_x, double ridge_y, double tol) {
//   Rcpp::stop("cpp_rkhs_xy_dense: not implemented in this build");
// }
// 
// // [[Rcpp::export]]
// SEXP cpp_kf_pls_stream(SEXP X_ptr, SEXP Y_ptr,
//                        int ncomp, int chunk_size,
//                        double process_var = 1e-4,
//                        double meas_var = 1e-3) {
//   Rcpp::stop("cpp_kf_pls_stream: not implemented in this build");
// }

// ---------- 1) Dense kernel-PLS from a precomputed Gram matrix --------------
//
// Minimal placeholder that returns correctly-shaped objects.
// Flesh out: center K (double-centering), run kernel SIMPLS in dual, etc.
// 
// [[Rcpp::export]]
SEXP cpp_kpls_from_gram(const arma::mat& K,
                        const arma::mat& Y,
                        int ncomp,
                        double tol = 1e-8,
                        bool compute_scores = false) {
  if (K.n_rows != K.n_cols) stop("K must be square (n x n)");
  if (Y.n_rows != K.n_rows) stop("nrow(Y) must match nrow(K)");
  if (ncomp < 0) ncomp = 0;
  
  const arma::uword n = K.n_rows;
  const arma::uword m = Y.n_cols;
  
  // stub shapes
  arma::mat alpha(n, m, arma::fill::zeros);     // dual coefficients
  arma::rowvec y_means = arma::mean(Y, 0);
  arma::mat scores;                              // optional
  if (compute_scores && ncomp > 0) {
    scores.set_size(n, (arma::uword)ncomp);
    scores.zeros();
  }
  
  // Return a kernel-friendly structure. Leave primal coefficients empty.
  return List::create(
    _["coefficients"] = R_NilValue,  // primal beta not set in kernel stub
    _["dual_coef"]    = alpha,       // n x m
    _["intercept"]    = as<NumericVector>(wrap(y_means)),
    _["x_weights"]    = R_NilValue,
    _["x_loadings"]   = R_NilValue,
    _["y_loadings"]   = R_NilValue,
    _["scores"]       = scores.n_elem ? wrap(scores) : R_NilValue,
    _["x_means"]      = R_NilValue,  // unknown without X
    _["y_means"]      = as<NumericVector>(wrap(y_means)),
    _["ncomp"]        = std::max(0, ncomp)
  );
}

// ---------- 2) Block Gram builder from a big.matrix (streaming-friendly) ----
//
// Computes a linear Gram block G = X[rows_i, ] %*% t(X[rows_j, ])
// Placeholder supports only "linear" kernel. Extend later with RBF etc.
//
// [[Rcpp::export]]
SEXP cpp_kernel_gram_block(SEXP X_ptr,
                           IntegerVector rows_i,
                           IntegerVector rows_j,
                           std::string kernel = "linear",
                           double gamma = 1.0,
                           double coef0 = 0.0,
                           double degree = 2.0) {
  if (kernel != "linear")
    warning("stub only implements kernel='linear' at the moment");
  
  XPtr<BigMatrix> xp(X_ptr);
  ensure_double_bigmatrix(*xp, "X");
  
  // pull the two row blocks
  NumericMatrix Ai = rows_from_bigmatrix(*xp, rows_i);
  NumericMatrix Aj = rows_from_bigmatrix(*xp, rows_j);
  
  arma::mat A( Ai.begin(), Ai.nrow(), Ai.ncol(), /*copy_aux_mem=*/false, /*strict=*/true );
  arma::mat B( Aj.begin(), Aj.nrow(), Aj.ncol(), /*copy_aux_mem=*/false, /*strict=*/true );
  
  arma::mat G = A * B.t(); // linear kernel
  return wrap(G);
}


