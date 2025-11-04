#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <algorithm>
#include <cmath>
#include <vector>

// [[Rcpp::plugins(cpp17)]]

namespace {

inline void ensure_double_matrix(const BigMatrix& mat, const char* name) {
  if (mat.matrix_type() != 8) {
    std::string msg = std::string(name) + " must be a double precision big.matrix";
    Rcpp::stop(msg);
  }
}
  
  inline arma::vec select_initial_u(const arma::mat& Y) {
    const arma::uword q = Y.n_cols;
    arma::uword chosen = 0;
    double best_var = -1.0;
    for (arma::uword j = 0; j < q; ++j) {
      double v = arma::var(Y.col(j));
      if (v > best_var) {
        best_var = v;
        chosen = j;
      }
    }
    arma::vec u = Y.col(chosen);
    if (best_var <= 0.0) {
      // fall back to the first column to preserve dimensions
      u.zeros();
    }
    return u;
  }
  
  inline bool is_converged(const arma::vec& current, const arma::vec& previous, double tol) {
    if (previous.n_elem == 0) {
      return false;
    }
    double norm_curr = arma::norm(current, 2);
    double norm_prev = arma::norm(previous, 2);
    double denom = std::max(1.0, std::max(norm_curr, norm_prev));
    double diff = arma::norm(current - previous, 2) / denom;
    return diff < tol;
  }

  inline void load_deflated_x_row(std::size_t row_idx,
                                  std::vector<double>& buffer,
                                  MatrixAccessor<double>& Xacc,
                                  const std::vector<double>& meanX,
                                  const std::vector<double>& scaleX,
                                  bool center,
                                  bool scale,
                                  const std::vector<arma::vec>& scores,
                                  const arma::mat& P,
                                  int current_components) {
    const std::size_t px = buffer.size();
    for (std::size_t j = 0; j < px; ++j) {
      double val = Xacc[j][row_idx];
      if (center) {
        val -= meanX[j];
      }
      if (scale) {
        val /= scaleX[j];
      }
      for (int h = 0; h < current_components; ++h) {
        val -= scores[h][row_idx] * P(j, h);
      }
      buffer[j] = val;
    }
  }
  
  inline void load_deflated_y_row(std::size_t row_idx,
                                  std::vector<double>& buffer,
                                  MatrixAccessor<double>& Yacc,
                                  const std::vector<double>& meanY,
                                  const std::vector<double>& scaleY,
                                  bool center,
                                  bool scale,
                                  const std::vector<arma::vec>& scores,
                                  const arma::mat& Q,
                                  const arma::vec& B,
                                  int current_components) {
    const std::size_t q = buffer.size();
    for (std::size_t k = 0; k < q; ++k) {
      double val = Yacc[k][row_idx];
      if (center) {
        val -= meanY[k];
      }
      if (scale) {
        val /= scaleY[k];
      }
      for (int h = 0; h < current_components; ++h) {
        val -= B[h] * scores[h][row_idx] * Q(k, h);
      }
      buffer[k] = val;
    }
  }
} // namespace

// [[Rcpp::export]]
Rcpp::List big_plsr_fit_nipals(SEXP X_ptr,
                               SEXP Y_ptr,
                               int ncomp,
                               bool center = true,
                               bool scale = false) {
  if (ncomp <= 0) {
    Rcpp::stop("ncomp must be positive");
  }
  
  Rcpp::XPtr<BigMatrix> xMat(X_ptr);
  Rcpp::XPtr<BigMatrix> yMat(Y_ptr);
  
  ensure_double_matrix(*xMat, "X");
  ensure_double_matrix(*yMat, "Y");
  
  const arma::uword n = xMat->nrow();
  const arma::uword px = xMat->ncol();
  const arma::uword q = yMat->ncol();
  
  if (yMat->nrow() != n) {
    Rcpp::stop("X and Y must have the same number of rows");
  }
  if (n == 0) {
    Rcpp::stop("Input matrices must contain at least one row");
  }
  
  arma::mat X(static_cast<double*>(xMat->matrix()), n, px, false, true);
  arma::mat Y(static_cast<double*>(yMat->matrix()), n, q, false, true);
  
  arma::rowvec meanX(px, arma::fill::zeros);
  arma::rowvec meanY(q, arma::fill::zeros);
  arma::rowvec scaleX(px, arma::fill::ones);
  arma::rowvec scaleY(q, arma::fill::ones);
  
  arma::mat Xc = X;
  arma::mat Yc = Y;
  
  if (center) {
    meanX = arma::mean(Xc, 0);
    meanY = arma::mean(Yc, 0);
    Xc.each_row() -= meanX;
    Yc.each_row() -= meanY;
  }
  
  if (scale) {
    scaleX = arma::stddev(Xc, 0, 0);
    scaleY = arma::stddev(Yc, 0, 0);
    scaleX.replace(0.0, 1.0);
    scaleY.replace(0.0, 1.0);
    Xc.each_row() /= scaleX;
    Yc.each_row() /= scaleY;
  }
  
  const int max_comp = std::min<int>(ncomp, static_cast<int>(px));
  const double tol = 1e-10;
  const int max_iter = 500;
  
  arma::mat W(px, max_comp, arma::fill::zeros);
  arma::mat P(px, max_comp, arma::fill::zeros);
  arma::mat Q(q, max_comp, arma::fill::zeros);
  arma::mat T(n, max_comp, arma::fill::zeros);
  arma::vec B(max_comp, arma::fill::zeros);
  
  arma::mat Xdef = Xc;
  arma::mat Ydef = Yc;
  
  int actual_comp = 0;
  
  for (int a = 0; a < max_comp; ++a) {
    arma::vec u = select_initial_u(Ydef);
    arma::vec u_prev;
    
    if (arma::norm(u, 2) <= 1e-12) {
      break;
    }
    
    arma::vec w(px, arma::fill::zeros);
    arma::vec t(n, arma::fill::zeros);
    arma::vec c(q, arma::fill::zeros);
    
    bool converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
      double u_norm_sq = arma::dot(u, u);
      if (u_norm_sq <= 1e-20) {
        break;
      }
      
      w = Xdef.t() * u / u_norm_sq;
      double w_norm = arma::norm(w, 2);
      if (w_norm <= 1e-12) {
        break;
      }
      w /= w_norm;
      
      t = Xdef * w;
      double t_norm_sq = arma::dot(t, t);
      if (t_norm_sq <= 1e-20) {
        break;
      }
      
      c = Ydef.t() * t / t_norm_sq;
      double c_norm_sq = arma::dot(c, c);
      if (c_norm_sq <= 1e-20) {
        break;
      }
      
      u_prev = u;
      u = Ydef * c / c_norm_sq;
      
      if (is_converged(u, u_prev, tol)) {
        converged = true;
        break;
      }
    }
    
    if (!converged) {
      double u_norm = arma::norm(u, 2);
      if (u_norm <= 1e-12) {
        break;
      }
    }
    
    double t_norm_sq = arma::dot(t, t);
    if (t_norm_sq <= 1e-20) {
      break;
    }
    
    arma::vec p = Xdef.t() * t / t_norm_sq;
    arma::vec qvec = c;
    
    double b = arma::dot(t, u) / t_norm_sq;
    
    Xdef -= t * p.t();
    Ydef -= b * t * qvec.t();
    
    W.col(a) = w;
    P.col(a) = p;
    Q.col(a) = qvec;
    T.col(a) = t;
    B[a] = b;
    ++actual_comp;
  }
  
  if (actual_comp == 0) {
    return Rcpp::List::create(
      Rcpp::Named("coefficients") = R_NilValue,
      Rcpp::Named("intercept") = R_NilValue,
      Rcpp::Named("weights") = R_NilValue,
      Rcpp::Named("loadings") = R_NilValue,
      Rcpp::Named("y_loadings") = R_NilValue,
      Rcpp::Named("scores") = R_NilValue,
      Rcpp::Named("x_means") = meanX,
      Rcpp::Named("y_means") = meanY,
      Rcpp::Named("x_scales") = scaleX,
      Rcpp::Named("y_scales") = scaleY
    );
  }
  
  arma::mat W_used = W.cols(0, actual_comp - 1);
  arma::mat P_used = P.cols(0, actual_comp - 1);
  arma::mat Q_used = Q.cols(0, actual_comp - 1);
  arma::mat T_used = T.cols(0, actual_comp - 1);
  arma::vec B_used = B.subvec(0, actual_comp - 1);
  
  arma::mat PtW = P_used.t() * W_used;
  arma::mat PtW_inv;
  bool status = arma::inv(PtW_inv, PtW);
  if (!status) {
    Rcpp::stop("Failed to invert P'W matrix in NIPALS solver");
  }
  
  arma::mat coef_internal = W_used * PtW_inv * arma::diagmat(B_used) * Q_used.t();
  
  if (scale) {
    arma::vec scaleX_vec = scaleX.t();
    arma::vec scaleY_vec = scaleY.t();
    scaleX_vec.transform([](double val) { return 1.0 / val; });
    arma::mat scaleXInv = arma::diagmat(scaleX_vec);
    arma::mat scaleYDiag = arma::diagmat(scaleY_vec);
    coef_internal = scaleXInv * coef_internal * scaleYDiag;
  }
  
  arma::mat scores = T_used;
  
  arma::rowvec intercept = meanY;
  if (center) {
    intercept = meanY - meanX * coef_internal;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef_internal,
    Rcpp::Named("intercept") = intercept,
    Rcpp::Named("weights") = W_used,
    Rcpp::Named("loadings") = P_used,
    Rcpp::Named("y_loadings") = Q_used,
    Rcpp::Named("scores") = scores,
    Rcpp::Named("x_means") = meanX,
    Rcpp::Named("y_means") = meanY,
    Rcpp::Named("x_scales") = scaleX,
    Rcpp::Named("y_scales") = scaleY
  );
}


// [[Rcpp::export]]
Rcpp::List big_plsr_stream_fit_nipals(SEXP X_ptr,
                                      SEXP Y_ptr,
                                      int ncomp,
                                      bool center = true,
                                      bool scale = false,
                                      std::size_t block_size = 1024) {
  if (ncomp <= 0) {
    Rcpp::stop("ncomp must be positive");
  }
  if (block_size == 0) {
    Rcpp::stop("block_size must be strictly positive");
  }
  
  Rcpp::XPtr<BigMatrix> xMat(X_ptr);
  Rcpp::XPtr<BigMatrix> yMat(Y_ptr);
  
  ensure_double_matrix(*xMat, "X");
  ensure_double_matrix(*yMat, "Y");
  
  const std::size_t n = xMat->nrow();
  const std::size_t px = xMat->ncol();
  const std::size_t q = yMat->ncol();
  
  if (yMat->nrow() != static_cast<int>(n)) {
    Rcpp::stop("X and Y must have the same number of rows");
  }
  if (n == 0) {
    Rcpp::stop("Input matrices must contain at least one row");
  }
  
  ncomp = std::min<int>(ncomp, static_cast<int>(px));
  
  MatrixAccessor<double> Xacc(*xMat);
  MatrixAccessor<double> Yacc(*yMat);
  
  std::vector<double> meanX(px, 0.0);
  std::vector<double> meanY(q, 0.0);
  
  if (center) {
    for (std::size_t j = 0; j < px; ++j) {
      const double* col = Xacc[j];
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        sum += col[i];
      }
      meanX[j] = sum / static_cast<double>(n);
    }
    for (std::size_t k = 0; k < q; ++k) {
      const double* col = Yacc[k];
      double sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        sum += col[i];
      }
      meanY[k] = sum / static_cast<double>(n);
    }
  }
  
  std::vector<double> scaleX(px, 1.0);
  std::vector<double> scaleY(q, 1.0);
  
  if (scale) {
    for (std::size_t j = 0; j < px; ++j) {
      const double* col = Xacc[j];
      double sumsq = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double val = col[i];
        if (center) {
          val -= meanX[j];
        }
        sumsq += val * val;
      }
      double denom = std::max<std::size_t>(1, n - 1);
      double s = std::sqrt(sumsq / static_cast<double>(denom));
      if (s > 0.0) {
        scaleX[j] = s;
      }
    }
    for (std::size_t k = 0; k < q; ++k) {
      const double* col = Yacc[k];
      double sumsq = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        double val = col[i];
        if (center) {
          val -= meanY[k];
        }
        sumsq += val * val;
      }
      double denom = std::max<std::size_t>(1, n - 1);
      double s = std::sqrt(sumsq / static_cast<double>(denom));
      if (s > 0.0) {
        scaleY[k] = s;
      }
    }
  }
  
  const double tol = 1e-10;
  const int max_iter = 500;
  
  arma::mat W(px, ncomp, arma::fill::zeros);
  arma::mat P(px, ncomp, arma::fill::zeros);
  arma::mat Q(q, ncomp, arma::fill::zeros);
  arma::vec B(ncomp, arma::fill::zeros);
  
  std::vector<arma::vec> scores;
  scores.reserve(ncomp);
  
  std::vector<double> x_buffer(px, 0.0);
  std::vector<double> y_buffer(q, 0.0);
  
  int actual_comp = 0;
  
  for (int a = 0; a < ncomp; ++a) {
    // Determine initial u vector using column of deflated Y with largest variance.
    std::vector<double> sumsqY(q, 0.0);
    for (std::size_t start = 0; start < n; start += block_size) {
      std::size_t end = std::min<std::size_t>(n, start + block_size);
      for (std::size_t i = start; i < end; ++i) {
        load_deflated_y_row(i, y_buffer, Yacc, meanY, scaleY, center, scale,
                            scores, Q, B, actual_comp);
        for (std::size_t k = 0; k < q; ++k) {
          double val = y_buffer[k];
          sumsqY[k] += val * val;
        }
      }
    }
    
    std::size_t chosen_col = 0;
    double best_var = -1.0;
    for (std::size_t k = 0; k < q; ++k) {
      if (sumsqY[k] > best_var) {
        best_var = sumsqY[k];
        chosen_col = k;
      }
    }
    
    if (best_var <= 1e-20) {
      break;
    }
    
    arma::vec u(n, arma::fill::zeros);
    for (std::size_t start = 0; start < n; start += block_size) {
      std::size_t end = std::min<std::size_t>(n, start + block_size);
      for (std::size_t i = start; i < end; ++i) {
        load_deflated_y_row(i, y_buffer, Yacc, meanY, scaleY, center, scale,
                            scores, Q, B, actual_comp);
        u[i] = y_buffer[chosen_col];
      }
    }
    
    arma::vec t(n, arma::fill::zeros);
    arma::vec c(q, arma::fill::zeros);
    arma::vec w(px, arma::fill::zeros);
    bool converged = false;
    
    for (int iter = 0; iter < max_iter; ++iter) {
      arma::vec u_prev = u;
      double u_norm_sq = arma::dot(u, u);
      if (u_norm_sq <= 1e-20) {
        converged = false;
        break;
      }
      
      w.zeros();
      for (std::size_t start = 0; start < n; start += block_size) {
        std::size_t end = std::min<std::size_t>(n, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
          load_deflated_x_row(i, x_buffer, Xacc, meanX, scaleX, center, scale,
                              scores, P, actual_comp);
          double ui = u[i];
          for (std::size_t j = 0; j < px; ++j) {
            w[j] += x_buffer[j] * ui;
          }
        }
      }
      
      w /= u_norm_sq;
      double w_norm = arma::norm(w, 2);
      if (w_norm <= 1e-12) {
        converged = false;
        break;
      }
      w /= w_norm;
      
      t.zeros();
      arma::vec q_acc(q, arma::fill::zeros);
      double t_norm_sq = 0.0;
      for (std::size_t start = 0; start < n; start += block_size) {
        std::size_t end = std::min<std::size_t>(n, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
          load_deflated_x_row(i, x_buffer, Xacc, meanX, scaleX, center, scale,
                              scores, P, actual_comp);
          double ti = 0.0;
          for (std::size_t j = 0; j < px; ++j) {
            ti += x_buffer[j] * w[j];
          }
          t[i] = ti;
          t_norm_sq += ti * ti;
          
          load_deflated_y_row(i, y_buffer, Yacc, meanY, scaleY, center, scale,
                              scores, Q, B, actual_comp);
          for (std::size_t k = 0; k < q; ++k) {
            q_acc[k] += y_buffer[k] * ti;
          }
        }
      }
      
      if (t_norm_sq <= 1e-20) {
        converged = false;
        break;
      }
      
      c = q_acc / t_norm_sq;
      double c_norm_sq = arma::dot(c, c);
      if (c_norm_sq <= 1e-20) {
        converged = false;
        break;
      }
      
      arma::vec u_new(n, arma::fill::zeros);
      double u_new_norm_sq = 0.0;
      for (std::size_t start = 0; start < n; start += block_size) {
        std::size_t end = std::min<std::size_t>(n, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
          load_deflated_y_row(i, y_buffer, Yacc, meanY, scaleY, center, scale,
                              scores, Q, B, actual_comp);
          double dot_val = 0.0;
          for (std::size_t k = 0; k < q; ++k) {
            dot_val += y_buffer[k] * c[k];
          }
          double ui_new = dot_val / c_norm_sq;
          u_new[i] = ui_new;
          u_new_norm_sq += ui_new * ui_new;
        }
      }
      
      double diff_norm = arma::norm(u_new - u_prev, 2);
      double denom = std::max(1.0, std::max(std::sqrt(u_new_norm_sq), arma::norm(u_prev, 2)));
      if (denom <= 0.0) {
        denom = 1.0;
      }
      
      u = u_new;
      
      if (diff_norm / denom < tol) {
        converged = true;
        break;
      }
    }
    
    if (!converged) {
      double u_norm = arma::norm(u, 2);
      if (u_norm <= 1e-12) {
        break;
      }
    }
    
    double t_norm_sq = arma::dot(t, t);
    if (t_norm_sq <= 1e-20) {
      break;
    }
    
    arma::vec pvec(px, arma::fill::zeros);
    for (std::size_t start = 0; start < n; start += block_size) {
      std::size_t end = std::min<std::size_t>(n, start + block_size);
      for (std::size_t i = start; i < end; ++i) {
        load_deflated_x_row(i, x_buffer, Xacc, meanX, scaleX, center, scale,
                            scores, P, actual_comp);
        double ti = t[i];
        for (std::size_t j = 0; j < px; ++j) {
          pvec[j] += x_buffer[j] * ti;
        }
      }
    }
    pvec /= t_norm_sq;
    
    arma::vec qvec = c;
    
    double b = arma::dot(t, u) / t_norm_sq;
    
    W.col(a) = w;
    P.col(a) = pvec;
    Q.col(a) = qvec;
    B[a] = b;
    scores.push_back(t);
    
    ++actual_comp;
  }
  
  if (actual_comp == 0) {
    return Rcpp::List::create(
      Rcpp::Named("coefficients") = R_NilValue,
      Rcpp::Named("intercept") = R_NilValue,
      Rcpp::Named("weights") = R_NilValue,
      Rcpp::Named("loadings") = R_NilValue,
      Rcpp::Named("y_loadings") = R_NilValue,
      Rcpp::Named("scores") = R_NilValue,
      Rcpp::Named("x_means") = Rcpp::NumericVector(meanX.begin(), meanX.end()),
      Rcpp::Named("y_means") = Rcpp::NumericVector(meanY.begin(), meanY.end()),
      Rcpp::Named("x_scales") = Rcpp::NumericVector(scaleX.begin(), scaleX.end()),
      Rcpp::Named("y_scales") = Rcpp::NumericVector(scaleY.begin(), scaleY.end())
    );
  }
  
  arma::mat W_used = W.cols(0, actual_comp - 1);
  arma::mat P_used = P.cols(0, actual_comp - 1);
  arma::mat Q_used = Q.cols(0, actual_comp - 1);
  arma::vec B_used = B.subvec(0, actual_comp - 1);
  
  arma::mat PtW = P_used.t() * W_used;
  arma::mat PtW_inv;
  bool status = arma::inv(PtW_inv, PtW);
  if (!status) {
    Rcpp::stop("Failed to invert P'W matrix in streaming NIPALS solver");
  }
  
  arma::mat coef_internal = W_used * PtW_inv * arma::diagmat(B_used) * Q_used.t();
  
  if (scale) {
    arma::vec scaleX_vec(px);
    arma::vec scaleY_vec(q);
    for (std::size_t j = 0; j < px; ++j) {
      scaleX_vec[j] = scaleX[j];
    }
    for (std::size_t k = 0; k < q; ++k) {
      scaleY_vec[k] = scaleY[k];
    }
    scaleX_vec.transform([](double val) { return 1.0 / val; });
    arma::mat scaleXInv = arma::diagmat(scaleX_vec);
    arma::mat scaleYDiag = arma::diagmat(scaleY_vec);
    coef_internal = scaleXInv * coef_internal * scaleYDiag;
  }
  
  arma::rowvec meanX_row(px);
  arma::rowvec meanY_row(q);
  for (std::size_t j = 0; j < px; ++j) {
    meanX_row[j] = meanX[j];
  }
  for (std::size_t k = 0; k < q; ++k) {
    meanY_row[k] = meanY[k];
  }
  
  arma::rowvec intercept = meanY_row;
  if (center) {
    intercept = meanY_row - meanX_row * coef_internal;
  }
  
  arma::mat scores_mat(n, actual_comp, arma::fill::zeros);
  for (int comp = 0; comp < actual_comp; ++comp) {
    scores_mat.col(comp) = scores[comp];
  }
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef_internal,
    Rcpp::Named("intercept") = intercept,
    Rcpp::Named("weights") = W_used,
    Rcpp::Named("loadings") = P_used,
    Rcpp::Named("y_loadings") = Q_used,
    Rcpp::Named("scores") = scores_mat,
    Rcpp::Named("x_means") = Rcpp::NumericVector(meanX.begin(), meanX.end()),
    Rcpp::Named("y_means") = Rcpp::NumericVector(meanY.begin(), meanY.end()),
    Rcpp::Named("x_scales") = Rcpp::NumericVector(scaleX.begin(), scaleX.end()),
    Rcpp::Named("y_scales") = Rcpp::NumericVector(scaleY.begin(), scaleY.end())
  );
}
