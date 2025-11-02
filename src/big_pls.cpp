#include <RcppArmadillo.h>
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
namespace {

inline void ensure_double_matrix(const BigMatrix& mat) {
  if (mat.matrix_type() != 8) {
    stop("big.matrix must be of type double");
  }
}

double compute_mean(const NumericVector& x) {
  double sum = 0.0;
  const R_xlen_t n = x.size();
  for (R_xlen_t i = 0; i < n; ++i) {
    sum += x[i];
  }
  return sum / static_cast<double>(n);
}

std::vector<double> compute_column_means(MatrixAccessor<double>& X,
                                         std::size_t nrows, std::size_t ncols,
                                         bool center) {
  std::vector<double> means(ncols, 0.0);
  if (!center) {
    return means;
  }
  for (std::size_t j = 0; j < ncols; ++j) {
    const double* col = X[j];
    double sum = 0.0;
    for (std::size_t i = 0; i < nrows; ++i) {
      sum += col[i];
    }
    means[j] = sum / static_cast<double>(nrows);
  }
  return means;
}

std::vector<double> compute_column_scales(MatrixAccessor<double>& X,
                                          const std::vector<double>& means,
                                          std::size_t nrows, std::size_t ncols,
                                          bool scale) {
  std::vector<double> scales(ncols, 1.0);
  if (!scale) {
    return scales;
  }
  for (std::size_t j = 0; j < ncols; ++j) {
    const double* col = X[j];
    double ss = 0.0;
    for (std::size_t i = 0; i < nrows; ++i) {
      const double centered = col[i] - means[j];
      ss += centered * centered;
    }
    const double denom = static_cast<double>(nrows > 1 ? (nrows - 1) : 1);
    double sd = std::sqrt(ss / denom);
    if (sd == 0.0 || !std::isfinite(sd)) {
      sd = 1.0;
    }
    scales[j] = sd;
  }
  return scales;
}

std::vector<double> initialize_y(const NumericVector& y,
                                 bool center, bool scale,
                                 double& mean_out, double& scale_out) {
  const std::size_t n = static_cast<std::size_t>(y.size());
  mean_out = 0.0;
  scale_out = 1.0;
  if (center) {
    mean_out = compute_mean(y);
  }
  std::vector<double> res(n);
  for (std::size_t i = 0; i < n; ++i) {
    res[i] = static_cast<double>(y[i]) - mean_out;
  }
  if (scale) {
    double ss = 0.0;
    for (double val : res) {
      ss += val * val;
    }
    const double denom = static_cast<double>(n > 1 ? (n - 1) : 1);
    scale_out = std::sqrt(ss / denom);
    if (scale_out == 0.0 || !std::isfinite(scale_out)) {
      scale_out = 1.0;
    }
    for (double& val : res) {
      val /= scale_out;
    }
  }
  return res;
}

List big_pls_core(SEXP x_ptr, const NumericVector& y, int ncomp,
                  bool center_x, bool scale_x,
                  bool center_y, bool scale_y,
                  int chunk_size) {
  XPtr<BigMatrix> xp(x_ptr);
  BigMatrix* bm = xp.get();
  ensure_double_matrix(*bm);
  MatrixAccessor<double> X(*bm);
  
  const std::size_t n = static_cast<std::size_t>(bm->nrow());
  const std::size_t p = static_cast<std::size_t>(bm->ncol());
  
  if (y.size() != static_cast<R_xlen_t>(n)) {
    stop("Response length does not match number of rows in X");
  }
  
  if (n == 0 || p == 0) {
    stop("X must have positive dimensions");
  }
  
  const std::size_t max_comp = std::min<std::size_t>({static_cast<std::size_t>(ncomp), n, p});
  if (max_comp == 0) {
    stop("Number of components must be positive");
  }
  
  std::vector<double> x_means = compute_column_means(X, n, p, center_x);
  std::vector<double> x_scales = compute_column_scales(X, x_means, n, p, scale_x);
  
  double y_mean = 0.0, y_scale = 1.0;
  std::vector<double> y_res = initialize_y(y, center_y, scale_y, y_mean, y_scale);
  
  std::vector<double> T_scores(n * max_comp, 0.0);
  std::vector<double> P_loadings(p * max_comp, 0.0);
  std::vector<double> W_weights(p * max_comp, 0.0);
  std::vector<double> Q_loadings(max_comp, 0.0);
  
  std::vector<double> w(p);
  std::vector<double> t(n);
  std::vector<double> pvec(p);
  
  const int effective_chunk = (chunk_size <= 0) ? static_cast<int>(n) : std::min<int>(chunk_size, static_cast<int>(n));
  
  for (std::size_t comp = 0; comp < max_comp; ++comp) {
    std::fill(w.begin(), w.end(), 0.0);
    
    for (std::size_t start = 0; start < n; start += effective_chunk) {
      const std::size_t end = std::min<std::size_t>(start + effective_chunk, n);
      for (std::size_t j = 0; j < p; ++j) {
        const double* col = X[j];
        double sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
          double xij = (col[i] - x_means[j]) / x_scales[j];
          if (comp > 0) {
            double correction = 0.0;
            for (std::size_t h = 0; h < comp; ++h) {
              correction += T_scores[i + h * n] * P_loadings[j + h * p];
            }
            xij -= correction;
          }
          sum += xij * y_res[i];
        }
        w[j] += sum;
      }
    }
    
    double norm_w = 0.0;
    for (double val : w) {
      norm_w += val * val;
    }
    norm_w = std::sqrt(norm_w);
    if (norm_w == 0.0 || !std::isfinite(norm_w)) {
      warning("Component %d resulted in zero weight vector; stopping early", comp + 1);
      ncomp = static_cast<int>(comp);
      break;
    }
    for (double& val : w) {
      val /= norm_w;
    }
    
    std::fill(t.begin(), t.end(), 0.0);
    for (std::size_t start = 0; start < n; start += effective_chunk) {
      const std::size_t end = std::min<std::size_t>(start + effective_chunk, n);
      for (std::size_t i = start; i < end; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
          double xij = (X[j][i] - x_means[j]) / x_scales[j];
          if (comp > 0) {
            double correction = 0.0;
            for (std::size_t h = 0; h < comp; ++h) {
              correction += T_scores[i + h * n] * P_loadings[j + h * p];
            }
            xij -= correction;
          }
          sum += xij * w[j];
        }
        t[i] += sum;
      }
    }
    
    double tt = 0.0;
    for (double val : t) {
      tt += val * val;
    }
    if (tt == 0.0 || !std::isfinite(tt)) {
      warning("Component %d resulted in zero score variance; stopping early", comp + 1);
      ncomp = static_cast<int>(comp);
      break;
    }
    
    std::fill(pvec.begin(), pvec.end(), 0.0);
    for (std::size_t start = 0; start < n; start += effective_chunk) {
      const std::size_t end = std::min<std::size_t>(start + effective_chunk, n);
      for (std::size_t j = 0; j < p; ++j) {
        const double* col = X[j];
        double sum = 0.0;
        for (std::size_t i = start; i < end; ++i) {
          double xij = (col[i] - x_means[j]) / x_scales[j];
          if (comp > 0) {
            double correction = 0.0;
            for (std::size_t h = 0; h < comp; ++h) {
              correction += T_scores[i + h * n] * P_loadings[j + h * p];
            }
            xij -= correction;
          }
          sum += xij * t[i];
        }
        pvec[j] += sum / tt;
      }
    }
    
    double q = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      q += y_res[i] * t[i];
    }
    q /= tt;
    
    for (std::size_t i = 0; i < n; ++i) {
      y_res[i] -= q * t[i];
    }
    
    for (std::size_t j = 0; j < p; ++j) {
      W_weights[j + comp * p] = w[j];
      P_loadings[j + comp * p] = pvec[j];
    }
    for (std::size_t i = 0; i < n; ++i) {
      T_scores[i + comp * n] = t[i];
    }
    Q_loadings[comp] = q;
  }
  
  const std::size_t used_comp = std::min<std::size_t>(max_comp, static_cast<std::size_t>(ncomp));
  
  arma::mat W_mat(p, used_comp, arma::fill::zeros);
  arma::mat P_mat(p, used_comp, arma::fill::zeros);
  arma::vec Q_vec(used_comp, arma::fill::zeros);
  
  for (std::size_t comp = 0; comp < used_comp; ++comp) {
    for (std::size_t j = 0; j < p; ++j) {
      W_mat(j, comp) = W_weights[j + comp * p];
      P_mat(j, comp) = P_loadings[j + comp * p];
    }
    Q_vec[comp] = Q_loadings[comp];
  }
  
  arma::vec coeff_scaled;
  if (used_comp > 0) {
    arma::mat R_mat = trans(P_mat) * W_mat;
    arma::vec inner = arma::solve(R_mat, Q_vec);
    coeff_scaled = W_mat * inner;
  } else {
    coeff_scaled = arma::vec(p, arma::fill::zeros);
  }
  
  arma::vec coeff_full(p, arma::fill::zeros);
  if (used_comp > 0) {
    coeff_full = coeff_scaled;
  }
  
  NumericVector coefficients(p);
  for (std::size_t j = 0; j < p; ++j) {
    double scale_factor = x_scales[j];
    if (scale_factor == 0.0 || !std::isfinite(scale_factor)) {
      scale_factor = 1.0;
    }
    double value = (used_comp > 0) ? coeff_full[j] : 0.0;
    value *= y_scale;
    value /= scale_factor;
    coefficients[j] = value;
  }
  
  double intercept = y_mean;
  for (std::size_t j = 0; j < p; ++j) {
    intercept -= coefficients[j] * x_means[j];
  }
  
  NumericMatrix scores(n, used_comp);
  NumericMatrix loadings(p, used_comp);
  NumericMatrix weights(p, used_comp);
  
  for (std::size_t comp = 0; comp < used_comp; ++comp) {
    for (std::size_t i = 0; i < n; ++i) {
      scores(i, comp) = T_scores[i + comp * n];
    }
    for (std::size_t j = 0; j < p; ++j) {
      loadings(j, comp) = P_loadings[j + comp * p];
      weights(j, comp) = W_weights[j + comp * p];
    }
  }
  
  return List::create(
    Named("coefficients") = coefficients,
    Named("intercept") = intercept,
    Named("scores") = scores,
    Named("loadings") = loadings,
    Named("weights") = weights,
    Named("y_mean") = y_mean,
    Named("y_scale") = y_scale,
    Named("x_means") = NumericVector(x_means.begin(), x_means.end()),
    Named("x_scales") = NumericVector(x_scales.begin(), x_scales.end()),
    Named("ncomp") = static_cast<int>(used_comp)
  );
}

} // namespace

// [[Rcpp::export]]
SEXP big_pls_fit_cpp(SEXP x_ptr, NumericVector y, int ncomp,
                     bool center_x, bool scale_x,
                     bool center_y, bool scale_y) {
  return big_pls_core(x_ptr, y, ncomp, center_x, scale_x, center_y, scale_y, -1);
}

// [[Rcpp::export]]
SEXP big_pls_stream_cpp(SEXP x_ptr, NumericVector y, int ncomp,
                        bool center_x, bool scale_x,
                        bool center_y, bool scale_y,
                        int chunk_size) {
  if (chunk_size <= 0) {
    stop("chunk_size must be positive for streaming fits");
  }
  return big_pls_core(x_ptr, y, ncomp, center_x, scale_x, center_y, scale_y, chunk_size);
}
