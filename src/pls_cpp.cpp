#include <Rcpp.h>
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
                          int max_iter) {
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
  
  NumericVector coefficients(p);
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
  
  NumericMatrix weights_mat = to_numeric_matrix(weights, p, ncomp, used_comp);
  NumericMatrix loadings_mat = to_numeric_matrix(loadings, p, ncomp, used_comp);
  NumericMatrix scores_mat(n, used_comp);
  for (int h = 0; h < used_comp; ++h) {
    for (std::size_t i = 0; i < n; ++i) {
      scores_mat(i, h) = scores[h * n + i];
    }
  }
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
    Named("coefficients") = coefficients,
    Named("intercept") = intercept,
    Named("x_center") = x_mean_vec,
    Named("x_scale") = x_scale_vec,
    Named("y_center") = center ? Rcpp::wrap(y_mean) : R_NilValue,
    Named("y_scale") = scale ? Rcpp::wrap(y_scale) : R_NilValue,
    Named("weights") = weights_mat,
    Named("loadings") = loadings_mat,
    Named("scores") = scores_mat,
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
                             double tol) {
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
  
  std::vector<double> XtX(p * p, 0.0);
  std::vector<double> Xty(p, 0.0);
  std::vector<double> row_buffer(p);
  const double* y_col = y_acc[0];
  
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    for (std::size_t i = start; i < end; ++i) {
      const double y_centered = y_col[i] - (center ? y_mean : 0.0);
      const double y_scaled = y_centered / (scale ? y_scale : 1.0);
      for (std::size_t j = 0; j < p; ++j) {
        const double* col = X_acc[j];
        const double centered = col[i] - (center ? x_means[j] : 0.0);
        const double scaled_val = centered / (scale ? x_scales[j] : 1.0);
        row_buffer[j] = scaled_val;
      }
      for (std::size_t j = 0; j < p; ++j) {
        Xty[j] += row_buffer[j] * y_scaled;
      }
      for (std::size_t j = 0; j < p; ++j) {
        const double xj = row_buffer[j];
        const std::size_t offset = j * p;
        for (std::size_t k = j; k < p; ++k) {
          const double val = xj * row_buffer[k];
          XtX[offset + k] += val;
          if (k != j) {
            XtX[k * p + j] += val;
          }
        }
      }
    }
  }
  
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
  
  NumericVector coefficients(p);
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
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    for (std::size_t i = start; i < end; ++i) {
      for (std::size_t j = 0; j < p; ++j) {
        const double* col = X_acc[j];
        const double centered = col[i] - (center ? x_means[j] : 0.0);
        const double scaled_val = centered / (scale ? x_scales[j] : 1.0);
        row_buffer[j] = scaled_val;
      }
      for (int h = 0; h < used_comp; ++h) {
        double val = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
          val += row_buffer[j] * W[j * ncomp + h];
        }
        scores_mat_vec[i * used_comp + h] = val;
      }
    }
  }
  
  NumericMatrix weights_mat = to_numeric_matrix(W, p, ncomp, used_comp);
  NumericMatrix loadings_mat = to_numeric_matrix(P, p, ncomp, used_comp);
  NumericMatrix scores_out(n, used_comp);
  for (std::size_t i = 0; i < n; ++i) {
    for (int h = 0; h < used_comp; ++h) {
      scores_out(i, h) = scores_mat_vec[i * used_comp + h];
    }
  }
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
    Named("coefficients") = coefficients,
    Named("intercept") = intercept,
    Named("x_center") = x_mean_vec,
    Named("x_scale") = x_scale_vec,
    Named("y_center") = center ? Rcpp::wrap(y_mean) : R_NilValue,
    Named("y_scale") = scale ? Rcpp::wrap(y_scale) : R_NilValue,
    Named("weights") = weights_mat,
    Named("loadings") = loadings_mat,
    Named("scores") = scores_out,
    Named("y_loadings") = q_out,
    Named("ncomp") = used_comp,
    Named("chunk_size") = static_cast<int>(chunk)
  );
}
