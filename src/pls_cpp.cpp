#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

// [[Rcpp::depends(bigmemory)]]
// [[Rcpp::plugins(cpp17)]]

using Rcpp::List;
using Rcpp::NumericMatrix;
using Rcpp::NumericVector;
using Rcpp::Named;
using Rcpp::XPtr;

#include "bigmatrix_utils.hpp"

namespace {

inline void check_double_matrix(const BigMatrix& mat, const char* name) {
  if (mat.matrix_type() != 8) {
    Rcpp::stop("%s must be a double-precision big.matrix", name);
  }
}
  
  inline double dot_product(const std::vector<double>& a,
                            const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
  }
  
  inline double norm2(const std::vector<double>& x) {
    return std::sqrt(dot_product(x, x));
  }
  
  inline void compute_means(BigMatrix& X,
                            std::vector<double>& means) {
    const std::size_t n = X.nrow();
    const std::size_t p = X.ncol();
    MatrixAccessor<double> accessor(X);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = accessor[j];
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        sum += col[i];
      }
      means[j] = sum / static_cast<double>(n);
    }
  }
  
  inline void compute_scales(BigMatrix& X,
                             const std::vector<double>& means,
                             std::vector<double>& scales,
                             bool center) {
    const std::size_t n = X.nrow();
    const std::size_t p = X.ncol();
    MatrixAccessor<double> accessor(X);
    const double denom = n > 1 ? static_cast<double>(n - 1) : 1.0;
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = accessor[j];
      double sum_sq = 0.0;
      const double mean = center ? means[j] : 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double diff = col[i] - mean;
        sum_sq += diff * diff;
      }
      double sd = std::sqrt(sum_sq / denom);
      if (sd <= 0.0) {
        sd = 1.0;
      }
      scales[j] = sd;
    }
  }
  
  inline void mat_vec_mul_sym(const std::vector<double>& mat,
                              const std::vector<double>& vec,
                              std::size_t dim,
                              std::vector<double>& out) {
    for (std::size_t i = 0; i < dim; ++i) {
      double total = 0.0;
      const std::size_t offset = i * dim;
      for (std::size_t j = 0; j < dim; ++j) {
        total += mat[offset + j] * vec[j];
      }
      out[i] = total;
    }
  }
  
  inline std::vector<double> solve_linear_system(std::vector<double> A,
                                                 std::vector<double> b,
                                                 int n,
                                                 double tol) {
    // Row-major A
    for (int i = 0; i < n; ++i) {
      int pivot = i;
      double max_val = std::abs(A[i * n + i]);
      for (int r = i + 1; r < n; ++r) {
        const double val = std::abs(A[r * n + i]);
        if (val > max_val) {
          max_val = val;
          pivot = r;
        }
      }
      if (max_val < tol) {
        Rcpp::stop("Singular system when computing regression coefficients");
      }
      if (pivot != i) {
        for (int c = 0; c < n; ++c) {
          std::swap(A[i * n + c], A[pivot * n + c]);
        }
        std::swap(b[i], b[pivot]);
      }
      const double diag = A[i * n + i];
      for (int c = i; c < n; ++c) {
        A[i * n + c] /= diag;
      }
      b[i] /= diag;
      for (int r = 0; r < n; ++r) {
        if (r == i) continue;
        const double factor = A[r * n + i];
        if (factor == 0.0) continue;
        for (int c = i; c < n; ++c) {
          A[r * n + c] -= factor * A[i * n + c];
        }
        b[r] -= factor * b[i];
      }
    }
    return b;
  }
  
  // inline double column_dot(const std::vector<double>& mat,
  //                          std::size_t rows,
  //                          std::size_t cols,
  //                          std::size_t col,
  //                          const std::vector<double>& vec) {
  //   double total = 0.0;
  //   for (std::size_t i = 0; i < rows; ++i) {
  //     total += mat[i * cols + col] * vec[i];
  //   }
  //   return total;
  // }
  
  inline void orthogonalize_against(std::vector<double>& vec,
                                    const std::vector<double>& basis,
                                    std::size_t rows,
                                    std::size_t cols,
                                    std::size_t upto) {
    for (std::size_t h = 0; h < upto; ++h) {
      double proj = 0.0;
      for (std::size_t i = 0; i < rows; ++i) {
        proj += basis[i * cols + h] * vec[i];
      }
      for (std::size_t i = 0; i < rows; ++i) {
        vec[i] -= basis[i * cols + h] * proj;
      }
    }
  }
  
  // inline NumericMatrix to_numeric_matrix(const std::vector<double>& mat,
  //                                        std::size_t rows,
  //                                        std::size_t cols) {
  //   NumericMatrix res(rows, cols);
  //   for (std::size_t j = 0; j < cols; ++j) {
  //     for (std::size_t i = 0; i < rows; ++i) {
  //       res(i, j) = mat[i * cols + j];
  //     }
  //   }
  //   return res;
  // }
  
  inline NumericMatrix to_numeric_matrix(const std::vector<double>& mat,
                                         std::size_t rows,
                                         std::size_t cols,
                                         std::size_t used_cols) {
    NumericMatrix res(rows, used_cols);
    for (std::size_t j = 0; j < used_cols; ++j) {
      for (std::size_t i = 0; i < rows; ++i) {
        res(i, j) = mat[i * cols + j];
      }
    }
    return res;
  }
  
} // namespace

// [[Rcpp::export]]
List pls_nipals_bigmemory(SEXP X_ptrSEXP,
                          SEXP y_ptrSEXP,
                          int ncomp,
                          bool center,
                          bool scale,
                          double tol,
                          int max_iter,
                          bool return_big) {
  if (ncomp <= 0) {
    Rcpp::stop("ncomp must be positive");
  }
  if (max_iter <= 0) {
    max_iter = 100;
  }
  (void)max_iter;
  
  XPtr<BigMatrix> X_ptr(X_ptrSEXP);
  XPtr<BigMatrix> y_ptr(y_ptrSEXP);
  check_double_matrix(*X_ptr, "X");
  check_double_matrix(*y_ptr, "y");
  
  const std::size_t n = static_cast<std::size_t>(X_ptr->nrow());
  const std::size_t p = static_cast<std::size_t>(X_ptr->ncol());
  const std::size_t y_cols = static_cast<std::size_t>(y_ptr->ncol());
  const std::size_t y_rows = static_cast<std::size_t>(y_ptr->nrow());
  if (y_cols != 1) {
    Rcpp::stop("y must have exactly one column");
  }
  if (y_rows != n) {
    Rcpp::stop("Dimensions of X and y do not match");
  }
  
  MatrixAccessor<double> X_acc(*X_ptr);
  MatrixAccessor<double> y_acc(*y_ptr);
  
  std::vector<double> x_means(center ? p : 0u, 0.0);
  std::vector<double> x_scales(scale ? p : 0u, 1.0);
  double y_mean = 0.0;
  double y_scale = 1.0;
  
  if (center) {
    compute_means(*X_ptr, x_means);
    const double* y_col = y_acc[0];
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      sum += y_col[i];
    }
    y_mean = sum / static_cast<double>(n);
  }
  
  if (scale) {
    compute_scales(*X_ptr, x_means, x_scales, center);
    const double* y_col = y_acc[0];
    const double mean = center ? y_mean : 0.0;
    double sum_sq = 0.0;
    const double denom = n > 1 ? static_cast<double>(n - 1) : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double diff = y_col[i] - mean;
      sum_sq += diff * diff;
    }
    y_scale = std::sqrt(sum_sq / denom);
    if (y_scale <= 0.0) {
      y_scale = 1.0;
    }
  }
  
  std::vector<double> y_res(n);
  {
    const double* y_col = y_acc[0];
    const double mean = center ? y_mean : 0.0;
    const double inv_scale = 1.0 / (scale ? y_scale : 1.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double centered = y_col[i] - mean;
      y_res[i] = centered * inv_scale;
    }
  }
  
  std::vector<double> x_inv_scale(p, 1.0);
  if (scale) {
    for (std::size_t j = 0; j < p; ++j) {
      x_inv_scale[j] = 1.0 / x_scales[j];
    }
  }
  
  std::vector<double> weights(p * ncomp, 0.0);
  std::vector<double> loadings(p * ncomp, 0.0);
  std::vector<double> scores(n * ncomp, 0.0);
  std::vector<double> t_vec(n);
  std::vector<double> w_vec(p);
  std::vector<double> p_vec(p);
  std::vector<double> y_loadings(ncomp, 0.0);
  
  int used_comp = 0;
  double y_denom_initial = dot_product(y_res, y_res);
  if (y_denom_initial < tol) {
    Rcpp::stop("Response has zero variance");
  }
  
  for (int h = 0; h < ncomp; ++h) {
    std::fill(w_vec.begin(), w_vec.end(), 0.0);
    const double y_denom = dot_product(y_res, y_res);
    if (y_denom < tol) {
      break;
    }
    
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      const double mean = center ? x_means[j] : 0.0;
      const double inv_scale = scale ? x_inv_scale[j] : 1.0;
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double value = (col[i] - mean) * inv_scale;
        for (int k = 0; k < h; ++k) {
          value -= scores[k * n + i] * loadings[j * ncomp + k];
        }
        sum += value * y_res[i];
      }
      w_vec[j] = sum / y_denom;
    }
    
    const double w_norm = norm2(w_vec);
    if (w_norm < tol) {
      break;
    }
    for (double& val : w_vec) {
      val /= w_norm;
    }
    
    std::fill(t_vec.begin(), t_vec.end(), 0.0);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      const double mean = center ? x_means[j] : 0.0;
      const double inv_scale = scale ? x_inv_scale[j] : 1.0;
      const double weight = w_vec[j];
      for (std::size_t i = 0; i < n; ++i) {
        double value = (col[i] - mean) * inv_scale;
        for (int k = 0; k < h; ++k) {
          value -= scores[k * n + i] * loadings[j * ncomp + k];
        }
        t_vec[i] += value * weight;
      }
    }
    
    const double t_norm_sq = dot_product(t_vec, t_vec);
    if (t_norm_sq < tol) {
      break;
    }
    
    for (std::size_t i = 0; i < n; ++i) {
      scores[h * n + i] = t_vec[i];
    }
    
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      const double mean = center ? x_means[j] : 0.0;
      const double inv_scale = scale ? x_inv_scale[j] : 1.0;
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double value = (col[i] - mean) * inv_scale;
        for (int k = 0; k < h; ++k) {
          value -= scores[k * n + i] * loadings[j * ncomp + k];
        }
        sum += value * t_vec[i];
      }
      const double p_val = sum / t_norm_sq;
      loadings[j * ncomp + h] = p_val;
      weights[j * ncomp + h] = w_vec[j];
    }
    
    double q = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      q += y_res[i] * t_vec[i];
    }
    q /= t_norm_sq;
    y_loadings[h] = q;
    
    for (std::size_t i = 0; i < n; ++i) {
      y_res[i] -= q * t_vec[i];
    }
    
    ++used_comp;
  }
  
  if (used_comp == 0) {
    Rcpp::stop("Unable to extract any latent components");
  }
  
  std::vector<double> PTW(used_comp * used_comp, 0.0);
  std::vector<double> q_vec(used_comp);
  for (int h = 0; h < used_comp; ++h) {
    for (int g = 0; g < used_comp; ++g) {
      double sum = 0.0;
      for (std::size_t j = 0; j < p; ++j) {
        sum += loadings[j * ncomp + h] * weights[j * ncomp + g];
      }
      PTW[h * used_comp + g] = sum;
    }
    q_vec[h] = y_loadings[h];
  }
  std::vector<double> temp = solve_linear_system(PTW, q_vec, used_comp, tol);
  
  std::vector<double> coefficients(p);
  for (std::size_t j = 0; j < p; ++j) {
    double beta = 0.0;
    for (int h = 0; h < used_comp; ++h) {
      beta += weights[j * ncomp + h] * temp[h];
    }
    const double scale_factor = scale ? x_scales[j] : 1.0;
    beta *= (scale ? y_scale : 1.0) / scale_factor;
    coefficients[j] = beta;
  }

  double intercept = center ? y_mean : 0.0;
  if (center) {
    double adj = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
      adj += coefficients[j] * x_means[j];
    }
    intercept -= adj;
  }
  
  Rcpp::RObject coefficients_out =
    make_vector_output(return_big, coefficients.data(), p, "coefficients");
  Rcpp::RObject loadings_out =
    make_matrix_output(return_big, loadings.data(), p, used_comp, "loadings");
  Rcpp::RObject scores_out =
    make_matrix_output(return_big, scores.data(), n, used_comp, "scores");
  
  NumericMatrix weights_mat = to_numeric_matrix(weights, p, ncomp, used_comp);

  NumericVector y_loadings_vec(used_comp);
  for (int h = 0; h < used_comp; ++h) {
    y_loadings_vec[h] = y_loadings[h];
  }
  
  NumericVector x_mean_vec(center ? p : 0);
  NumericVector x_scale_vec(scale ? p : 0);
  if (center) {
    for (std::size_t j = 0; j < p; ++j) {
      x_mean_vec[j] = x_means[j];
    }
  }
  if (scale) {
    for (std::size_t j = 0; j < p; ++j) {
      x_scale_vec[j] = x_scales[j];
    }
  }
  
  return List::create(
    Named("coefficients") = coefficients_out,
    Named("intercept") = intercept,
    Named("x_center") = x_mean_vec,
    Named("x_scale") = x_scale_vec,
    Named("y_center") = center ? Rcpp::wrap(y_mean) : R_NilValue,
    Named("y_scale") = scale ? Rcpp::wrap(y_scale) : R_NilValue,
    Named("weights") = weights_mat,
    Named("loadings") = loadings_out,
    Named("scores") = scores_out,
    Named("y_loadings") = y_loadings_vec,
    Named("ncomp") = used_comp
  );
}

// [[Rcpp::export]]
List pls_streaming_bigmemory(SEXP X_ptrSEXP,
                             SEXP y_ptrSEXP,
                             int ncomp,
                             int chunk_size,
                             bool center,
                             bool scale,
                             double tol,
                             bool return_big) {
  if (ncomp <= 0) {
    Rcpp::stop("ncomp must be positive");
  }
  XPtr<BigMatrix> X_ptr(X_ptrSEXP);
  XPtr<BigMatrix> y_ptr(y_ptrSEXP);
  check_double_matrix(*X_ptr, "X");
  check_double_matrix(*y_ptr, "y");
  
  const std::size_t n = static_cast<std::size_t>(X_ptr->nrow());
  const std::size_t p = static_cast<std::size_t>(X_ptr->ncol());
  const std::size_t y_cols = static_cast<std::size_t>(y_ptr->ncol());
  const std::size_t y_rows = static_cast<std::size_t>(y_ptr->nrow());
  if (y_cols != 1) {
    Rcpp::stop("y must have exactly one column");
  }
  if (y_rows != n) {
    Rcpp::stop("Dimensions of X and y do not match");
  }
  
  std::size_t chunk = chunk_size > 0
  ? static_cast<std::size_t>(chunk_size)
    : std::min<std::size_t>(n, static_cast<std::size_t>(1024));
  if (chunk == 0) {
    chunk = 1;
  }
  
  MatrixAccessor<double> X_acc(*X_ptr);
  MatrixAccessor<double> y_acc(*y_ptr);
  
  std::vector<double> x_means(center ? p : 0u, 0.0);
  double y_mean = 0.0;
  if (center) {
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        sum += col[i];
      }
      x_means[j] = sum / static_cast<double>(n);
    }
    const double* y_col = y_acc[0];
    double sum_y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      sum_y += y_col[i];
    }
    y_mean = sum_y / static_cast<double>(n);
  }
  
  std::vector<double> x_scales(scale ? p : 0u, 1.0);
  double y_scale = 1.0;
  if (scale) {
    const double denom = n > 1 ? static_cast<double>(n - 1) : 1.0;
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      const double mean = center ? x_means[j] : 0.0;
      double sum_sq = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const double diff = col[i] - mean;
        sum_sq += diff * diff;
      }
      double sd = std::sqrt(sum_sq / denom);
      if (sd <= 0.0) {
        sd = 1.0;
      }
      x_scales[j] = sd;
    }
    const double* y_col = y_acc[0];
    const double mean = center ? y_mean : 0.0;
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double diff = y_col[i] - mean;
      sum_sq += diff * diff;
    }
    y_scale = std::sqrt(sum_sq / (n > 1 ? static_cast<double>(n - 1) : 1.0));
    if (y_scale <= 0.0) {
      y_scale = 1.0;
    }
  }
  
  
  arma::mat XtX_mat(p, p, arma::fill::zeros);
  arma::vec Xty_vec(p, arma::fill::zeros);

  std::vector<double> row_buffer(p);

// Chunked accumulation using BLAS: XtX += B.t() * B; Xty += B.t() * y_chunk
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    const std::size_t m = end - start;
  
    // Build B (m x p) from big.matrix columns
    arma::mat B(m, p);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      // copy slice [start, end)
      std::memcpy(B.colptr(j), col + start, m * sizeof(double));
  }

  // Center/scale
  if (center) {
    arma::rowvec mu(p);
    for (std::size_t j = 0; j < p; ++j) mu[j] = x_means[j];
    B.each_row() -= mu;
  }
  if (scale) {
    arma::rowvec invs(p);
    for (std::size_t j = 0; j < p; ++j) invs[j] = 1.0 / x_scales[j];
    B.each_row() %= invs;
  }

  // y chunk
  arma::vec y_chunk(m);
  const double* ycol = y_acc[0];
  std::memcpy(y_chunk.memptr(), ycol + start, m * sizeof(double));
  if (center) y_chunk -= y_mean;
  if (scale)  y_chunk /= y_scale;

  // Accumulate
  XtX_mat += B.t() * B;
  Xty_vec += B.t() * y_chunk;
}

  std::vector<double> Xty(p); std::memcpy(Xty.data(), Xty_vec.memptr(), p*sizeof(double));
  std::vector<double> XtX(p*p);
  std::memcpy(XtX.data(), XtX_mat.memptr(), p*p*sizeof(double));

  std::vector<double> S = Xty;
  std::vector<double> W(p * ncomp, 0.0);
  std::vector<double> P(p * ncomp, 0.0);
  std::vector<double> V(p * ncomp, 0.0);
  std::vector<double> q_vec(ncomp, 0.0);
  std::vector<double> tmp_vec(p, 0.0);
  
  int used_comp = 0;
  for (int h = 0; h < ncomp; ++h) {
    std::vector<double> r = S;
    for (int j = 0; j < h; ++j) {
      std::vector<double> v_prev(p);
      for (std::size_t i = 0; i < p; ++i) {
        v_prev[i] = V[i * ncomp + j];
      }
      double proj = dot_product(v_prev, S);
      for (std::size_t i = 0; i < p; ++i) {
        r[i] -= v_prev[i] * proj;
      }
    }
    const double norm_r = norm2(r);
    if (norm_r < tol) {
      break;
    }
    for (double& val : r) {
      val /= norm_r;
    }
    
    mat_vec_mul_sym(XtX, r, p, tmp_vec);
    const double t_var = dot_product(r, tmp_vec);
    if (t_var < tol) {
      break;
    }
    
    std::vector<double> p_vec(p);
    for (std::size_t i = 0; i < p; ++i) {
      p_vec[i] = tmp_vec[i] / t_var;
      P[i * ncomp + h] = p_vec[i];
      W[i * ncomp + h] = r[i];
    }
    
    std::vector<double> v = p_vec;
    orthogonalize_against(v, V, p, ncomp, h);
    const double norm_v = norm2(v);
    if (norm_v < tol) {
      break;
    }
    for (std::size_t i = 0; i < p; ++i) {
      V[i * ncomp + h] = v[i] / norm_v;
    }
    
    const double q = dot_product(r, S) / t_var;
    q_vec[h] = q;
    for (std::size_t i = 0; i < p; ++i) {
      S[i] -= p_vec[i] * q;
    }
    ++used_comp;
  }
  
  if (used_comp == 0) {
    Rcpp::stop("Unable to extract any latent components");
  }
  
  std::vector<double> PTW(used_comp * used_comp, 0.0);
  for (int h = 0; h < used_comp; ++h) {
    for (int g = 0; g < used_comp; ++g) {
      double sum = 0.0;
      for (std::size_t j = 0; j < p; ++j) {
        sum += P[j * ncomp + h] * W[j * ncomp + g];
      }
      PTW[h * used_comp + g] = sum;
    }
  }
  std::vector<double> q_used(q_vec.begin(), q_vec.begin() + used_comp);
  std::vector<double> temp = solve_linear_system(PTW, q_used, used_comp, tol);
  
  std::vector<double> coefficients(p);
  for (std::size_t j = 0; j < p; ++j) {
    double beta = 0.0;
    for (int h = 0; h < used_comp; ++h) {
      beta += W[j * ncomp + h] * temp[h];
    }
    const double scale_factor = scale ? x_scales[j] : 1.0;
    beta *= (scale ? y_scale : 1.0) / scale_factor;
    coefficients[j] = beta;
  }
  
  double intercept = center ? y_mean : 0.0;
  if (center) {
    double adj = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
      adj += coefficients[j] * x_means[j];
    }
    intercept -= adj;
  }
  
  std::vector<double> scores_mat_vec(n * used_comp, 0.0);
  
// Streamed scores: optionally write to big.matrix when return_big is true
arma::mat Wmat(p, used_comp);
for (int h = 0; h < used_comp; ++h) {
  for (std::size_t j = 0; j < p; ++j) Wmat(j, h) = W[j * ncomp + h];
}

// Prepare big.matrix accessor if needed
Rcpp::S4 scores_bm;
MatrixAccessor<double> scores_acc_dummy(*X_ptr); // placeholder
bool use_big_scores = return_big;
if (use_big_scores) {
  scores_bm = allocate_big_matrix(n, used_comp, "scores");
}

for (std::size_t start = 0; start < n; start += chunk) {
  const std::size_t end = std::min<std::size_t>(n, start + chunk);
  const std::size_t m = end - start;

  arma::mat B(m, p);
  for (std::size_t j = 0; j < p; ++j) {
    const double* col = X_acc[j];
    std::memcpy(B.colptr(j), col + start, m * sizeof(double));
  }
  if (center) {
    arma::rowvec mu(p);
    for (std::size_t j = 0; j < p; ++j) mu[j] = x_means[j];
    B.each_row() -= mu;
  }
  if (scale) {
    arma::rowvec invs(p);
    for (std::size_t j = 0; j < p; ++j) invs[j] = 1.0 / x_scales[j];
    B.each_row() %= invs;
  }

  arma::mat S = B * Wmat; // (m x used_comp)

  if (use_big_scores) {
    // Lazy-init accessor to allocated big.matrix
    if (start == 0) {
      // extract BigMatrix* from S4
      Rcpp::XPtr<BigMatrix> sbm(scores_bm.slot("address"));
      scores_acc_dummy = MatrixAccessor<double>(*sbm);
    }
    for (int h = 0; h < used_comp; ++h) {
      std::memcpy(scores_acc_dummy[h] + start, S.colptr(h), m * sizeof(double));
    }
  } else {
    // Write into in-memory scores vector
    for (int h = 0; h < used_comp; ++h) {
      std::memcpy(&scores_mat_vec[start * used_comp + h], S.colptr(h), m * sizeof(double));
      // The above memcpy writes interleaved; safer to loop rows:
      for (std::size_t i = 0; i < m; ++i) {
        scores_mat_vec[(start + i) * used_comp + h] = S(i, h);
      }
    }
  }
}

Rcpp::RObject scores_out;
if (use_big_scores) {
  scores_out = scores_bm;
} else {
  scores_out = make_matrix_output(false, scores_mat_vec.data(), n, used_comp, "scores");
}

  
  Rcpp::RObject coefficients_out =
    make_vector_output(return_big, coefficients.data(), p, "coefficients");
  Rcpp::RObject loadings_out =
    make_matrix_output(return_big, P.data(), p, used_comp, "loadings");

  scores_out = use_big_scores ? Rcpp::RObject(scores_bm)
    : make_matrix_output(false, scores_mat_vec.data(),
      n, used_comp, "scores");
  
  NumericMatrix weights_mat = to_numeric_matrix(W, p, ncomp, used_comp);

  NumericVector q_out(used_comp);
  for (int h = 0; h < used_comp; ++h) {
    q_out[h] = q_vec[h];
  }
  
  NumericVector x_mean_vec(center ? p : 0);
  NumericVector x_scale_vec(scale ? p : 0);
  if (center) {
    for (std::size_t j = 0; j < p; ++j) {
      x_mean_vec[j] = x_means[j];
    }
  }
  if (scale) {
    for (std::size_t j = 0; j < p; ++j) {
      x_scale_vec[j] = x_scales[j];
    }
  }
  
  return List::create(
    Named("coefficients") = coefficients_out,
    Named("intercept") = intercept,
    Named("x_center") = x_mean_vec,
    Named("x_scale") = x_scale_vec,
    Named("y_center") = center ? Rcpp::wrap(y_mean) : R_NilValue,
    Named("y_scale") = scale ? Rcpp::wrap(y_scale) : R_NilValue,
    Named("weights") = weights_mat,
    Named("loadings") = loadings_out,
    Named("scores") = scores_out,
    Named("y_loadings") = q_out,
    Named("ncomp") = used_comp,
    Named("chunk_size") = static_cast<int>(chunk)
  );
}



#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
// [[Rcpp::plugins(cpp17)]]

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

#include <cstring>    // memcpy
#include <memory>     // shared_ptr
#include <algorithm>  // std::min
#include <functional> // std::function

using namespace Rcpp;
// using bigmemory::BigMatrix;
// using bigmemory::MatrixAccessor;

// [[Rcpp::export]]
SEXP cpp_big_pls_stream_fit_sink(SEXP X_ptrSEXP, SEXP y_ptrSEXP,
                                 SEXP scores_sinkSEXP, // NULL or big.matrix S4
                                 int ncomp, std::size_t chunk_size, double tol,
                                 bool return_big) {
  // ---- Attach X/Y ----
  XPtr<BigMatrix> X_ptr(X_ptrSEXP);
  XPtr<BigMatrix> y_ptr(y_ptrSEXP);
  MatrixAccessor<double> X_acc(*X_ptr);
  MatrixAccessor<double> y_acc(*y_ptr);
  
  const std::size_t n = static_cast<std::size_t>(X_ptr->nrow());
  const std::size_t p = static_cast<std::size_t>(X_ptr->ncol());
  if (y_ptr->ncol() != 1 || static_cast<std::size_t>(y_ptr->nrow()) != n)
    stop("y must be (n x 1)");
  
  // ---- Get the model (no scores) via the package R wrapper ----
  Environment ns = Environment::namespace_env("bigPLSR");
  Function fit_fun = ns["cpp_big_pls_stream_fit"];  // R wrapper that calls _bigPLSR_cpp_big_pls_stream_fit
  RObject fit_obj  = fit_fun(X_ptrSEXP, y_ptrSEXP, wrap(ncomp), wrap(chunk_size), wrap(tol), wrap(false));
  List fit = as<List>(fit_obj);
  
  const int used_comp = as<int>(fit["ncomp"]);
  if (used_comp <= 0) return fit;
  
  arma::mat W = fit.containsElementNamed("x_weights") ? as<arma::mat>(fit["x_weights"])
    : as<arma::mat>(fit["weights"]);
  
  // Optional center/scale (if provided by your fit)
  arma::rowvec mu, invs;
  bool do_center = false, do_scale = false;
  if (fit.containsElementNamed("x_means")) {
    NumericVector xmu = fit["x_means"];
    mu.set_size(p);
    for (std::size_t j = 0; j < p; ++j) mu[j] = xmu[j];
    do_center = true;
  }
  if (fit.containsElementNamed("x_scales")) {
    NumericVector xs = fit["x_scales"];
    invs.set_size(p);
    for (std::size_t j = 0; j < p; ++j) invs[j] = 1.0 / xs[j];
    do_scale = true;
  }
  
  // ---- Decide sink + build a single writer lambda ----
  const std::size_t chunk = (chunk_size > 0) ? chunk_size
  : std::min<std::size_t>(n, (std::size_t)4096);
  
  std::function<void(std::size_t /*start*/, const arma::mat& /*S*/)> write_scores;
  RObject scores_out;           // set for big.matrix cases
  bool to_inmemory_matrix = false;
  std::unique_ptr<arma::mat> T; // only used if writing to in-memory scores
  
  if (!Rf_isNull(scores_sinkSEXP)) {
    // External file-backed sink provided from R
    S4 sink_s4(scores_sinkSEXP);
    XPtr<BigMatrix> sink_ptr(sink_s4.slot("address"));
    if ((std::size_t)sink_ptr->nrow() != n ||
        (std::size_t)sink_ptr->ncol() != (std::size_t)used_comp)
      stop("scores sink dimension mismatch");
    
    // Capture a shared accessor by value in the lambda
    auto acc = std::make_shared<MatrixAccessor<double>>(*sink_ptr);
    write_scores = [acc, used_comp](std::size_t start, const arma::mat& S) {
      const std::size_t m = S.n_rows;
      for (int h = 0; h < used_comp; ++h) {
        std::memcpy((*acc)[h] + start, S.colptr(h), m * sizeof(double));
      }
    };
    scores_out = sink_s4; // finalize after loop
  } else if (return_big) {
    // Allocate an internal big.matrix sink
    S4 alloc_s4 = allocate_big_matrix(n, used_comp, "scores");
    XPtr<BigMatrix> alloc_ptr(alloc_s4.slot("address"));
    auto acc = std::make_shared<MatrixAccessor<double>>(*alloc_ptr);
    
    write_scores = [acc, used_comp](std::size_t start, const arma::mat& S) {
      const std::size_t m = S.n_rows;
      for (int h = 0; h < used_comp; ++h) {
        std::memcpy((*acc)[h] + start, S.colptr(h), m * sizeof(double));
      }
    };
    scores_out = alloc_s4; // finalize after loop
  } else {
    // In-memory R matrix sink
    T.reset(new arma::mat(n, used_comp, arma::fill::zeros));
    write_scores = [T_ptr = T.get()](std::size_t start, const arma::mat& S) {
      const std::size_t m = S.n_rows;
      T_ptr->rows(start, start + m - 1) = S;
    };
    to_inmemory_matrix = true;
  }
  
  // ---- Single chunk loop, same for all sinks ----
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    const std::size_t m   = end - start;
    
    arma::mat B(m, p);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      std::memcpy(B.colptr(j), col + start, m * sizeof(double));
    }
    if (do_center) B.each_row() -= mu;
    if (do_scale)  B.each_row() %= invs;
    
    arma::mat S = B * W;           // (m x used_comp), BLAS-backed
    write_scores(start, S);        // write to the chosen sink
  }
  
  // ---- Finalize return ----
  if (to_inmemory_matrix) {
    fit["scores"] = *T;
  } else {
    fit["scores"] = scores_out;    // S4 big.matrix
  }
  return fit;
}
