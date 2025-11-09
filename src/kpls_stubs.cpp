#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;

// ---------- helpers ---------------------------------------------------------

[[maybe_unused]] static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) {
    stop(std::string(nm) + " must be a double big.matrix");
  }
}

// pull a row subset (1-based indices from R) from a BigMatrix into an R matrix
[[maybe_unused]] static inline NumericMatrix rows_from_bigmatrix(BigMatrix& BM, const IntegerVector& idx_1based){
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
// SEXP cpp_kf_pls_stream(SEXP X_ptr, SEXP Y_ptr,
//                        int ncomp, int chunk_size,
//                        double process_var = 1e-4,
//                        double meas_var = 1e-3) {
//   Rcpp::stop("cpp_kf_pls_stream: not implemented in this build");
// }




