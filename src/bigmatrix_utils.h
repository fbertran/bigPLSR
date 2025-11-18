#ifndef BIGPLSR_BIGMATRIX_UTILS_HPP
#define BIGPLSR_BIGMATRIX_UTILS_HPP

#include <Rcpp.h>
#include <bigmemory/BigMatrix.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

inline int safe_matrix_dim(std::size_t value, const char* name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    Rcpp::stop(std::string("Dimension of ") + name + " exceeds supported range for big.matrix");
  }
  return static_cast<int>(value);
}

inline Rcpp::S4 allocate_big_matrix(std::size_t nrow,
                                    std::size_t ncol,
                                    const char* name) {
  static Rcpp::Function big_matrix =
    Rcpp::Environment::namespace_env("bigmemory")["big.matrix"];
  return big_matrix(Rcpp::Named("nrow") = safe_matrix_dim(nrow, name),
                    Rcpp::Named("ncol") = safe_matrix_dim(ncol, name),
                    Rcpp::Named("type") = "double");
}

inline void copy_column_major(const Rcpp::S4& target,
                              const double* source,
                              std::size_t nrow,
                              std::size_t ncol) {
  if (!source || nrow == 0 || ncol == 0) {
    return;
  }
  Rcpp::XPtr<BigMatrix> xp(target.slot("address"));
  double* dest = static_cast<double*>(xp->matrix());
  const std::size_t stride = nrow;
  for (std::size_t col = 0; col < ncol; ++col) {
    std::memcpy(dest + col * stride, source + col * stride, nrow * sizeof(double));
  }
}

inline Rcpp::RObject make_matrix_output(bool return_big,
                                        const double* data,
                                        std::size_t nrow,
                                        std::size_t ncol,
                                        const char* name) {
  if (nrow == 0 || ncol == 0) {
    return Rcpp::NumericMatrix(nrow, ncol);
  }
  if (return_big) {
    Rcpp::S4 bm = allocate_big_matrix(nrow, ncol, name);
    copy_column_major(bm, data, nrow, ncol);
    return Rcpp::RObject(static_cast<SEXP>(bm));   // explicit SEXP cast
//    return bm;                         // S4 -> SEXP
    }
  Rcpp::NumericMatrix mat(nrow, ncol);
  std::copy(data, data + (nrow * ncol), mat.begin());
  return Rcpp::RObject(mat);                        // or static_cast<SEXP>(mat)
//  std::copy(data, data + static_cast<size_t>(nrow) * static_cast<size_t>(ncol), mat.begin());
//  return mat;                          // NumericMatrix -> SEXP
}

inline Rcpp::RObject make_vector_output(bool return_big,
                                        const double* data,
                                        std::size_t length,
                                        const char* name) {
  if (length == 0) {
    return Rcpp::NumericVector(0);
  }
  if (return_big) {
    Rcpp::S4 bm = allocate_big_matrix(length, 1, name);
    copy_column_major(bm, data, length, 1);
    return Rcpp::RObject(static_cast<SEXP>(bm));   // explicit SEXP cast
//    return bm;
  }
  Rcpp::NumericVector vec(length);
  std::copy(data, data + length, vec.begin());
  return Rcpp::RObject(vec);                        // or static_cast<SEXP>(vec)
  //  std::copy(data, data + static_cast<size_t>(nrow) * static_cast<size_t>(ncol), vec.begin());
  //  return vec;                          // NumericMatrix -> SEXP
}

#endif  // BIGPLSR_BIGMATRIX_UTILS_HPP
