
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
// [[Rcpp::plugins(cpp17)]]

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

#include <algorithm>
#include <vector>
#include <string>

#include "bigmatrix_utils.hpp"

using namespace Rcpp;
using namespace arma;

// Core solver from centered cross-products (small p x p state)
static
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
    if (!std::isfinite(norm_w) || norm_w <= tol) break;
    w /= norm_w;

    arma::vec XtXw = XtX_curr * w;
    const double denom = arma::dot(w, XtXw);
    if (!std::isfinite(denom) || denom <= tol) break;
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

  arma::mat W_sub = (actual_components>0) ? W.cols(0, actual_components-1) : arma::mat(W.n_rows, 0);
  arma::mat P_sub = (actual_components>0) ? P.cols(0, actual_components-1) : arma::mat(P.n_rows, 0);
  arma::vec C_sub = (actual_components>0) ? C.head(actual_components) : arma::vec();

  arma::vec beta(W_sub.n_rows, arma::fill::zeros);
  double intercept = y_mean;
  if (actual_components>0) {
    arma::mat R = trans(P_sub) * W_sub;
    beta = W_sub * arma::solve(R, C_sub);
    intercept = y_mean - arma::dot(beta, x_mean);
  }

  return List::create(
    Named("coefficients") = beta,
    Named("intercept")    = intercept,
    Named("x_weights")    = W_sub,
    Named("x_loadings")   = P_sub,
    Named("y_loadings")   = C_sub,
    Named("x_means")      = x_mean,
    Named("y_mean")       = y_mean,
    Named("ncomp")        = actual_components
  );
}

// [[Rcpp::export]]
SEXP cpp_dense_pls_fit(Rcpp::NumericMatrix X, Rcpp::NumericVector y,
                       int ncomp, double tol,
                       bool compute_scores = false,
                       bool scores_big = false,
                       std::string scores_name = "scores") {
  const std::size_t n = X.nrow();
  const std::size_t p = X.ncol();
  if (y.size() != static_cast<R_xlen_t>(n))
    Rcpp::stop("y length must match nrow(X)");
  if (ncomp <= 0) Rcpp::stop("ncomp must be positive");
  if (static_cast<std::size_t>(ncomp) > p) ncomp = static_cast<int>(p);

  // Armadillo views (no copy)
  arma::mat Xa(X.begin(), n, p, /*copy_aux_mem*/ false, /*strict*/ true);
  arma::vec ya(y.begin(), n, /*copy_aux_mem*/ false, /*strict*/ true);

  arma::rowvec mean_row = arma::mean(Xa, 0);
  arma::vec x_mean = mean_row.t();
  double y_mean = arma::mean(ya);

  arma::mat Xc = Xa;           // copy once, then center
  Xc.each_row() -= mean_row;
  arma::vec yc = ya - y_mean;

  arma::mat XtX = Xc.t() * Xc;
  arma::vec XtY = Xc.t() * yc;

  List fit = solve_pls_from_cross(XtX, XtY, x_mean, y_mean, ncomp, tol);

  const int ncomp_out = Rcpp::as<int>(fit["ncomp"]);
  if (!compute_scores || ncomp_out == 0) return fit;
  
  // Build scores: T = Xc * W
  arma::mat W = fit["x_weights"];
  arma::mat T = Xc * W;

  if (!scores_big) {
    fit["scores"] = T;
    return fit;
  } else {
    // allocate big.matrix and copy column-major
    Rcpp::S4 bm = allocate_big_matrix(n, W.n_cols, scores_name.c_str());
    // Copy column-major
    // Ensure Armadillo memory is contiguous
    arma::mat Tc = T; // contiguous copy
    copy_column_major(bm, Tc.memptr(), n, W.n_cols);
    fit["scores"] = bm;
    return fit;
  }
}
