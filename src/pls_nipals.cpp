#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <algorithm>

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
