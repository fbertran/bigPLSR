#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, BH, bigmemory)]]

using namespace Rcpp;
using namespace arma;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

// [[Rcpp::plugins(cpp17)]]

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <vector>

namespace {

// use BigMatrix directly now
inline void ensure_double_matrix(const BigMatrix& mat, const std::string& name) {
  if (mat.matrix_type() != 8) Rcpp::stop(name + " must be a double-precision big.matrix");
}
  
  List solve_pls_from_cross(const arma::mat& XtX_center,
                            const arma::vec& XtY_center,
                            const arma::vec& x_mean,
                            double y_mean,
                            int ncomp,
                            double tol);
  
  // signatures no longer mention bigmemory::
  List pls_fit_dense(const BigMatrix& xMat,
                     const BigMatrix& yMat,
                     int ncomp,
                     double tol) {
    ensure_double_matrix(xMat, "x");
    ensure_double_matrix(yMat, "y");
    
    const std::size_t n = xMat.nrow();
    const std::size_t p = xMat.ncol();
    if (yMat.ncol() != 1) Rcpp::stop("y must have a single column");
    if (ncomp <= 0) Rcpp::stop("ncomp must be positive");
    if (static_cast<std::size_t>(ncomp) > p) ncomp = static_cast<int>(p);
    
    MatrixAccessor<double> xAcc(const_cast<BigMatrix&>(xMat));  // accessor must be non-const
    MatrixAccessor<double> yAcc(const_cast<BigMatrix&>(yMat));
    
    arma::mat X(n, p);
    arma::vec y(n);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col_ptr = xAcc[j];
      for (std::size_t i = 0; i < n; ++i) X(i, j) = col_ptr[i];
    }
    const double* y_ptr = yAcc[0];
    for (std::size_t i = 0; i < n; ++i) y[i] = y_ptr[i];
    
    arma::vec x_mean = arma::mean(X, 0).t();
    double y_mean = arma::mean(y);
    X.each_row() -= x_mean.t();
    arma::vec y_center = y - y_mean;
    
    arma::mat XtX = X.t() * X;
    arma::vec XtY = X.t() * y_center;
    
    return solve_pls_from_cross(XtX, XtY, x_mean, y_mean, ncomp, tol);
  }
  
  List pls_fit_streaming(const BigMatrix& xMat,
                         const BigMatrix& yMat,
                         int ncomp,
                         std::size_t chunk_size,
                         double tol) {
  ensure_double_matrix(xMat, "x");
  ensure_double_matrix(yMat, "y");
  const std::size_t n = xMat.nrow();
  const std::size_t p = xMat.ncol();
  if (yMat.ncol() != 1) {
    throw std::invalid_argument("y must have a single column");
  }
  if (chunk_size == 0) {
    throw std::invalid_argument("chunk_size must be greater than zero");
  }
  if (ncomp <= 0) {
    throw std::invalid_argument("ncomp must be positive");
  }
  if (static_cast<std::size_t>(ncomp) > p) {
    ncomp = static_cast<int>(p);
  }
  
  MatrixAccessor<double> xAcc(const_cast<BigMatrix&>(xMat));  // accessor must be non-const
  MatrixAccessor<double> yAcc(const_cast<BigMatrix&>(yMat));

  arma::vec sumX(p, arma::fill::zeros);
  arma::mat sumXX(p, p, arma::fill::zeros);
  arma::vec sumXY(p, arma::fill::zeros);
  double sumY = 0.0;
  std::size_t processed = 0;
  std::vector<double> row_buffer(p);
  
  for (std::size_t start = 0; start < n; start += chunk_size) {
    std::size_t end = std::min(start + chunk_size, n);
    for (std::size_t i = start; i < end; ++i) {
      const double y_val = yAcc[0][i];
      sumY += y_val;
      ++processed;
      for (std::size_t j = 0; j < p; ++j) {
        const double x_val = xAcc[j][i];
        row_buffer[j] = x_val;
        sumX[j] += x_val;
      }
      for (std::size_t j = 0; j < p; ++j) {
        const double xj = row_buffer[j];
        sumXY[j] += xj * y_val;
        for (std::size_t k = j; k < p; ++k) {
          const double prod = xj * row_buffer[k];
          sumXX(j, k) += prod;
          if (k != j) {
            sumXX(k, j) += prod;
          }
        }
      }
    }
  }
  
  if (processed == 0) {
    throw std::invalid_argument("no observations available for streaming PLS fit");
  }
  
  const double n_obs = static_cast<double>(processed);
  arma::vec x_mean = sumX / n_obs;
  double y_mean = sumY / n_obs;
  arma::mat XtX = sumXX - n_obs * (x_mean * x_mean.t());
  arma::vec XtY = sumXY - n_obs * x_mean * y_mean;
  
  return solve_pls_from_cross(XtX, XtY, x_mean, y_mean, ncomp, tol);
}

// Core solver that consumes centered cross-products.
List solve_pls_from_cross(const arma::mat& XtX_center,
                          const arma::vec& XtY_center,
                          const arma::vec& x_mean,
                          double y_mean,
                          int ncomp,
                          double tol) {
  const std::size_t p = XtX_center.n_rows;
  arma::mat XtX_curr = XtX_center;
  arma::vec XtY_curr = XtY_center;
  
  arma::mat W(p, ncomp, arma::fill::zeros);
  arma::mat P(p, ncomp, arma::fill::zeros);
  arma::vec C(ncomp, arma::fill::zeros);
  
  int actual_components = 0;
  for (int h = 0; h < ncomp; ++h) {
    arma::vec w = XtY_curr;
    const double norm_w = std::sqrt(arma::dot(w, w));
    if (!std::isfinite(norm_w) || norm_w <= tol) {
      break;
    }
    w /= norm_w;
    
    arma::vec XtXw = XtX_curr * w;
    const double denom = arma::dot(w, XtXw);
    if (!std::isfinite(denom) || denom <= tol) {
      break;
    }
    const double wXtY = arma::dot(w, XtY_curr);
    const double c_val = wXtY / denom;
    arma::vec p_vec = XtXw / denom;
    
    W.col(h) = w;
    P.col(h) = p_vec;
    C[h] = c_val;
    
    arma::mat XtX_new = XtX_curr - XtXw * p_vec.t() - p_vec * XtXw.t() + denom * (p_vec * p_vec.t());
    arma::vec XtY_new = XtY_curr - XtXw * c_val - p_vec * wXtY + denom * p_vec * c_val;
    
    XtX_curr = 0.5 * (XtX_new + XtX_new.t());
    XtY_curr = XtY_new;
    ++actual_components;
  }
  
  if (actual_components == 0) {
    return List::create(Named("coefficients") = arma::vec(p, arma::fill::zeros),
                        Named("intercept") = y_mean,
                        Named("x_weights") = arma::mat(p, 0),
                        Named("x_loadings") = arma::mat(p, 0),
                        Named("y_loadings") = arma::mat(1, 0),
                        Named("x_means") = x_mean,
                        Named("y_mean") = y_mean,
                        Named("ncomp") = 0);
  }
  
  arma::mat W_sub = W.cols(0, actual_components - 1);
  arma::mat P_sub = P.cols(0, actual_components - 1);
  arma::vec C_sub = C.head(actual_components);
  arma::mat R = trans(P_sub) * W_sub;
  arma::vec beta = W_sub * arma::solve(R, C_sub);
  const double intercept = y_mean - arma::dot(beta, x_mean);
  
  return List::create(Named("coefficients") = beta,
                      Named("intercept") = intercept,
                      Named("x_weights") = W_sub,
                      Named("x_loadings") = P_sub,
                      Named("y_loadings") = C_sub,
                      Named("x_means") = x_mean,
                      Named("y_mean") = y_mean,
                      Named("ncomp") = actual_components);
}

} // namespace

// [[Rcpp::export]]
SEXP cpp_big_pls_fit(SEXP x_ptr, SEXP y_ptr, int ncomp, double tol) {
  Rcpp::XPtr<BigMatrix> xMat(x_ptr);
  Rcpp::XPtr<BigMatrix> yMat(y_ptr);
  return pls_fit_dense(*xMat, *yMat, ncomp, tol);
}

// [[Rcpp::export]]

SEXP cpp_big_pls_stream_fit(SEXP x_ptr, SEXP y_ptr, int ncomp, std::size_t chunk_size, double tol) {
  Rcpp::XPtr<BigMatrix> xMat(x_ptr);
  Rcpp::XPtr<BigMatrix> yMat(y_ptr);
  return pls_fit_streaming(*xMat, *yMat, ncomp, chunk_size, tol);
}
