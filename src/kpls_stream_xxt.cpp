#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace Rcpp;

static inline void ensure_double_matrix(const BigMatrix& M, const char* name) {
  if (M.matrix_type() != 8) stop(std::string(name) + " must be a double-precision big.matrix");
}

// [[Rcpp::export]]
Rcpp::List cpp_kpls_stream_xxt(SEXP X_ptr, SEXP Y_ptr,
                               int ncomp,
                               int chunk_rows,
                               int chunk_cols,
                               bool center,
                               bool return_big) {
  if (ncomp <= 0) stop("ncomp must be positive");
  if (chunk_rows <= 0) chunk_rows = 8192;
  if (chunk_cols <= 0) chunk_cols = 64;
  
  Rcpp::XPtr<BigMatrix> Xp(X_ptr);
  Rcpp::XPtr<BigMatrix> Yp(Y_ptr);
  ensure_double_matrix(*Xp, "X");
  ensure_double_matrix(*Yp, "Y");
  
  const std::size_t n = Xp->nrow();
  const std::size_t p = Xp->ncol();
  const std::size_t m = Yp->ncol();
  if (Yp->nrow() != (index_type)n) stop("X and Y must have same number of rows");
  if (n == 0 || p == 0 || m == 0) stop("empty matrices");
  
  MatrixAccessor<double> Xacc(*Xp);
  MatrixAccessor<double> Yacc(*Yp);
  
  arma::rowvec meanX(p, arma::fill::zeros);
  arma::rowvec meanY(m, arma::fill::zeros);
  if (center) {
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = Xacc[j];
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += col[i];
      meanX[j] = s / double(n);
    }
    for (std::size_t k = 0; k < m; ++k) {
      const double* col = Yacc[k];
      double s = 0.0;
      for (std::size_t i = 0; i < n; ++i) s += col[i];
      meanY[k] = s / double(n);
    }
  }
  
  const int H = std::min<int>(ncomp, std::min<std::size_t>(n, p));
  arma::mat W(p, H, arma::fill::zeros);
  arma::mat P(p, H, arma::fill::zeros);
  arma::mat Q(m, H, arma::fill::zeros);
  arma::mat T(n, H, arma::fill::zeros);
  
  arma::mat Yres(n, m, arma::fill::zeros);
  for (std::size_t k = 0; k < m; ++k) {
    const double* col = Yacc[k];
    for (std::size_t i = 0; i < n; ++i) {
      double v = col[i];
      if (center) v -= meanY[k];
      Yres(i, k) = v;
    }
  }
  
  arma::vec u = Yres.col(0);
  int used = 0;
  
  std::vector<double> a(p, 0.0);
  
  for (int h = 0; h < H; ++h) {
    if (!u.is_finite() || arma::norm(u, 2) <= 1e-14) break;
    
    std::fill(a.begin(), a.end(), 0.0);
    for (std::size_t r0 = 0; r0 < n; r0 += (std::size_t)chunk_rows) {
      const std::size_t r1 = std::min<std::size_t>(n, r0 + (std::size_t)chunk_rows);
      for (std::size_t j = 0; j < p; ++j) {
        const double* xcol = Xacc[j];
        double acc = 0.0;
        for (std::size_t i = r0; i < r1; ++i) {
          double xv = xcol[i];
          if (center) xv -= meanX[j];
          acc += xv * u[i];
        }
        a[j] += acc;
      }
    }
    
    double a_norm = 0.0;
    for (std::size_t j = 0; j < p; ++j) a_norm += a[j]*a[j];
    a_norm = std::sqrt(a_norm);
    if (!(a_norm > 0)) break;
    for (std::size_t j = 0; j < p; ++j) a[j] /= a_norm;
    for (std::size_t j = 0; j < p; ++j) W(j, h) = a[j];
    
    std::vector<double> p_acc(p, 0.0);
    double t_norm2 = 0.0;
    for (std::size_t r0 = 0; r0 < n; r0 += (std::size_t)chunk_rows) {
      const std::size_t r1 = std::min<std::size_t>(n, r0 + (std::size_t)chunk_rows);
      for (std::size_t i = r0; i < r1; ++i) {
        double ti = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
          double xv = Xacc[j][i];
          if (center) xv -= meanX[j];
          ti += xv * a[j];
        }
        T(i, h) = ti;
        t_norm2 += ti*ti;
      }
      for (std::size_t j = 0; j < p; ++j) {
        const double* xcol = Xacc[j];
        double acc = 0.0;
        for (std::size_t i = r0; i < r1; ++i) {
          double xv = xcol[i];
          if (center) xv -= meanX[j];
          acc += xv * T(i, h);
        }
        p_acc[j] += acc;
      }
    }
    if (!(t_norm2 > 0)) break;
    for (std::size_t j = 0; j < p; ++j) P(j, h) = p_acc[j] / t_norm2;
    
    arma::rowvec q(m, arma::fill::zeros);
    for (std::size_t k = 0; k < m; ++k) {
      double acc = 0.0;
      for (std::size_t i = 0; i < n; ++i) acc += Yres(i, k) * T(i, h);
      q[k] = acc / t_norm2;
    }
    Q.col(h) = q.t();
    
    for (std::size_t i = 0; i < n; ++i) {
      const double ti = T(i, h);
      for (std::size_t k = 0; k < m; ++k) Yres(i, k) -= ti * q[k];
    }
    u = Yres * q.t();
    ++used;
  }
  
  if (used == 0) {
    return Rcpp::List::create(
      _["coefficients"] = R_NilValue,
      _["intercept"]    = R_NilValue,
      _["x_weights"]    = R_NilValue,
      _["x_loadings"]   = R_NilValue,
      _["y_loadings"]   = R_NilValue,
      _["scores"]       = R_NilValue,
      _["x_means"]      = meanX,
      _["y_means"]      = meanY,
      _["ncomp"]        = 0
    );
  }
  
  arma::mat W_used = W.cols(0, used-1);
  arma::mat P_used = P.cols(0, used-1);
  arma::mat Q_used = Q.cols(0, used-1);
  arma::mat T_used = T.cols(0, used-1);
  arma::mat R = P_used.t() * W_used;
  arma::mat Rinv; bool ok = arma::inv(Rinv, R);
  arma::mat beta = ok ? (W_used * Rinv * Q_used.t()) : arma::mat(p, Yp->ncol(), arma::fill::zeros);
  arma::rowvec intercept = meanY - meanX * beta;
  
  return Rcpp::List::create(
    _["coefficients"] = beta,
    _["intercept"]    = Rcpp::NumericVector(intercept.begin(), intercept.end()),
    _["x_weights"]    = W_used,
    _["x_loadings"]   = P_used,
    _["y_loadings"]   = Q_used,
    _["scores"]       = T_used,
    _["x_means"]      = Rcpp::NumericVector(meanX.begin(), meanX.end()),
    _["y_means"]      = Rcpp::NumericVector(meanY.begin(), meanY.end()),
    _["ncomp"]        = used
  );
}