#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <numeric>

#include "bigmatrix_utils.h"   // <- use package helpers (make_matrix_output, etc.)

// [[Rcpp::plugins(cpp17)]]

/*
 ======= RKHS PLS (Rosipal & Trejo, 2001) – Dense from Gram =======
 
 Inputs:
 K : n x n (not necessarily centered)
 Y : n x m (numeric matrix or vector)
 Output:
 List with:
 - dual_alpha  : n x m  (for prediction: Ŷ = Kc * dual_alpha + 1 * y_means)
 - intercept   : length m
 - scores      : n x A  (T)
 - y_loadings  : m x A  (Q)
 - ncomp       : A actually used
 - gram_center : logical TRUE (for clarity)
 */
[[maybe_unused]] static inline void center_inplace(arma::mat& M) {
  const arma::rowvec cm = arma::mean(M, 0);
  M.each_row() -= cm;
}

// Double-center Gram: Kc = H K H
[[maybe_unused]] static inline arma::mat center_gram(const arma::mat& K) {
  arma::mat Kc = K;
  arma::rowvec colm = arma::mean(K, 0);
  arma::colvec rowm = arma::mean(K, 1);
  const double g = arma::mean(colm);
  Kc.each_row() -= colm;
  Kc.each_col() -= rowm;
  Kc += g;
  return Kc;
}

[[maybe_unused]] inline void ensure_double_matrix(const BigMatrix& mat, const char* name) {
  if (mat.matrix_type() != 8) {
    stop(std::string(name) + " must be a double precision big.matrix");
  }
}
  
inline arma::vec select_initial_u(const arma::mat& Y) {
  const arma::uword q = Y.n_cols;
  arma::uword chosen = 0;
  double best_var = -1.0;
  for (arma::uword j = 0; j < q; ++j) {
    double v = arma::var(Y.col(j));
    if (v > best_var) { best_var = v; chosen = j; }
  }
  arma::vec u = Y.col(chosen);
  if (best_var <= 0.0) u.zeros();
  return u;
}

inline bool is_converged(const arma::vec& current, const arma::vec& previous, double tol) {
  if (previous.n_elem == 0) return false;
  double norm_curr = arma::norm(current, 2);
  double norm_prev = arma::norm(previous, 2);
  double denom = std::max(1.0, std::max(norm_curr, norm_prev));
  double diff  = arma::norm(current - previous, 2) / denom;
  return diff < tol;
}

// Y-centering helper (dense)
inline void center_Y_dense(arma::mat& Y, arma::rowvec& y_mean) {
  y_mean = arma::mean(Y, 0);
  Y.each_row() -= y_mean;
}

// Dense centering of a symmetric Gram matrix: Kc = K - r*1' - 1*c + mu
inline arma::mat center_gram_dense(arma::mat K) {
  const arma::uword n = K.n_rows;
  arma::vec r       = arma::mean(K, 1);
  arma::rowvec c    = arma::mean(K, 0);
  const double mu   = arma::accu(K) / (double(n) * double(n));
  K.each_col() -= r;
  K.each_row() -= c;
  K += mu;  // add scalar to all entries
  return K;
}

// --- Kernel helpers placed BEFORE any usage (fixes 'make_kernel' not found) ---
namespace {
inline arma::mat kernel_linear(const arma::mat& A, const arma::mat& B) {
  return A * B.t();
}
  inline arma::mat kernel_poly(const arma::mat& A, const arma::mat& B,
                               double gamma, int degree, double c0) {
    arma::mat G = A * B.t();
    if (gamma != 0.0) G *= gamma;
    if (c0    != 0.0) G += c0;
    arma::mat K(G.n_rows, G.n_cols, arma::fill::ones);
    for (int d = 0; d < degree; ++d) K %= G;   // elementwise power
    return K;
  }
  inline arma::mat kernel_rbf(const arma::mat& A, const arma::mat& B, double gamma) {
    // exp(-gamma ||a - b||^2) = exp(-gamma (||a||^2 + ||b||^2 - 2 a.b))
    arma::vec an = arma::sum(arma::square(A), 1);
    arma::vec bn = arma::sum(arma::square(B), 1);
    arma::mat G  = A * B.t();
    arma::mat D  = arma::repmat(an, 1, B.n_rows) + arma::repmat(bn.t(), A.n_rows, 1) - 2.0 * G;
    return arma::exp(-gamma * D);
  }
  inline arma::mat kernel_tanh(const arma::mat& A, const arma::mat& B,
                               double gamma, double c0) {
    arma::mat G = A * B.t();
    if (gamma != 0.0) G *= gamma;
    if (c0    != 0.0) G += c0;
    return arma::tanh(G);
  }
  inline arma::mat make_kernel(const arma::mat& A, const arma::mat& B,
                               const std::string& kernel,
                               double gamma, int degree, double c0) {
    if (kernel == "linear")                return kernel_linear(A,B);
    if (kernel == "rbf"   || kernel=="gaussian") return kernel_rbf(A,B,gamma);
    if (kernel == "poly"  || kernel=="polynomial") return kernel_poly(A,B,gamma,degree,c0);
    if (kernel == "tanh"  || kernel=="sigmoid")    return kernel_tanh(A,B,gamma,c0);
    Rcpp::stop("Unknown kernel type: " + kernel);
  }
} // anonymous namespace




// Kernel block builder (dense X)
inline arma::mat kernel_block_dense(const arma::mat& X,
                                    arma::uword r0, arma::uword r1,
                                    arma::uword c0, arma::uword c1,
                                    const std::string& kernel,
                                    double gamma, int degree, double coef0) {
  arma::mat Xr = X.rows(r0, r1 - 1);
  arma::mat Xc = X.rows(c0, c1 - 1);
  // Delegate to the unified kernel builder (supports linear / rbf / poly / tanh).
  return make_kernel(Xr, Xc, kernel, gamma, degree, coef0);
}

// Kernel block builder (bigmemory X)
inline arma::mat kernel_block_bigmem(BigMatrix& Xbm,
                                     arma::uword r0, arma::uword r1,
                                     arma::uword c0, arma::uword c1,
                                     const std::string& kernel,
                                     double gamma, int degree, double coef0) {
  ensure_double_matrix(Xbm, "X");
  MatrixAccessor<double> Xacc(Xbm);
  const arma::uword p  = Xbm.ncol();
  const arma::uword nr = r1 - r0;
  const arma::uword nc = c1 - c0;
  arma::mat Xr(nr, p, arma::fill::zeros);
  arma::mat Xc(nc, p, arma::fill::zeros);
  for (arma::uword j = 0; j < p; ++j) {
    for (arma::uword i = 0; i < nr; ++i) Xr(i,j) = Xacc[j][r0 + i];
    for (arma::uword i = 0; i < nc; ++i) Xc(i,j) = Xacc[j][c0 + i];
  }
  // Build the block via unified kernel routine.
  return make_kernel(Xr, Xc, kernel, gamma, degree, coef0);
}


/*
 ======= Block Gram multiply: V = K_c * U  (no full K) =======
 
 Computes V = H K(X,X) H U   with row-chunks on X and multiple columns of U.
 
 Inputs:
 X_ptr     : external pointer to bigmemory::BigMatrix (double)
 U         : n x r  (one or many right-hand sides)
 kernel    : "rbf" | "poly" | "linear" | "tanh"
 gamma     : kernel parameter (rbf/poly/tanh)
 degree    : poly degree
 coef0     : poly/tanh bias
 chunk_rows: block size for row loops
 center    : if true, apply H on both sides: Kc U = H K (H U)
 
 Returns:
 V : n x r   (numeric matrix)
 */


// --------------------------------------------------------------------------
// 1) Export: kernel Gram BLOCK (dense or bigmemory)
// --------------------------------------------------------------------------
// [[Rcpp::export]]
arma::mat cpp_kernel_gram_block(SEXP X,
                                arma::uword r0, arma::uword r1,
                                arma::uword c0, arma::uword c1,
                                std::string kernel,
                                double gamma, int degree, double coef0) {
  if (Rf_isMatrix(X)) {
    Rcpp::NumericMatrix Xm(X);
    arma::mat Xa(Xm.begin(), Xm.nrow(), Xm.ncol(), false, true);
    return kernel_block_dense(Xa, r0, r1, c0, c1, kernel, gamma, degree, coef0);
  }
  if (TYPEOF(X) == EXTPTRSXP) {
    Rcpp::XPtr<BigMatrix> Xptr(X);
    return kernel_block_bigmem(*Xptr, r0, r1, c0, c1, kernel, gamma, degree, coef0);
  }
  stop("cpp_kernel_gram_block: X must be numeric matrix or big.matrix external pointer");
}


// --------------------------------------------------------------------------
// 2) Export: KPLS from a *centered* Gram (dense)
//     - K is n x n, centered (HKH)
//     - Y is centered internally
//     - returns dual coefficients alpha (n x m), intercept = mean(Y), scores
// --------------------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List cpp_kpls_from_gram(Rcpp::NumericMatrix K_,
                              Rcpp::RObject Y_,
                              int ncomp,
                              double tol) {
  arma::mat K(K_.begin(), K_.nrow(), K_.ncol(), false, true);
  if (K.n_rows != K.n_cols) stop("Gram matrix must be square");
  const arma::uword n = K.n_rows;
  arma::mat Y;
  if (Rf_isMatrix(Y_)) {
    Rcpp::NumericMatrix Ym(Y_);
    if ((arma::uword)Ym.nrow() != n) stop("Y and K have incompatible rows");
    Y = arma::mat(Ym.begin(), Ym.nrow(), Ym.ncol(), false, true);
  } else {
    Rcpp::NumericVector yv(Y_);
    if ((arma::uword)yv.size() != n) stop("y and K have incompatible lengths");
    Y = arma::mat(yv.begin(), n, 1, false, true);
  }
  // Center Y
  arma::rowvec y_mean = arma::mean(Y, 0);
  arma::mat    Yc     = Y.each_row() - y_mean;
  
  ncomp = std::min<int>(ncomp, (int)n);
  arma::mat T(n, ncomp, arma::fill::zeros);
  std::vector<arma::vec> Ucols; Ucols.reserve(ncomp);
  arma::mat Yres = Yc;
  int used = 0;
  for (int a = 0; a < ncomp; ++a) {
    arma::vec u = select_initial_u(Yres);
    arma::vec t(n, arma::fill::zeros), t_prev;
    bool ok = true;
    for (int it = 0; it < 1000; ++it) {
      arma::vec Ku = K * u;
      double tnorm = arma::norm(Ku, 2);
      if (!std::isfinite(tnorm) || tnorm <= tol) { ok = false; break; }
      t_prev = t;  t = Ku / tnorm;
      arma::vec c = Yres.t() * t / arma::dot(t,t);
      double cn2 = arma::dot(c,c);
      if (!std::isfinite(cn2) || cn2 <= tol) { ok = false; break; }
      arma::vec u_new = Yres * c / cn2;
      double rel = (t_prev.n_elem == t.n_elem && t_prev.n_elem > 0)
        ? arma::norm(t - t_prev, 2) / std::max(1.0, arma::norm(t_prev, 2))
          : arma::norm(t, 2);
      u = u_new;
      if (rel <= tol) break;
    }
    if (!ok) break;
    arma::vec c = Yres.t() * t / arma::dot(t,t);
    Yres -= t * c.t();
    T.col(a) = t;
    arma::vec Ku = K * u;
    double tnorm = arma::norm(Ku, 2);
    if (tnorm <= tol) break;
    Ucols.push_back(u / tnorm);  // T = K * Ueff
    used++;
  }
  if (used == 0) {
    return List::create(
      _["coefficients"] = R_NilValue,
      _["intercept"]    = y_mean,
      _["scores"]       = R_NilValue,
      _["y_means"]      = y_mean,
      _["ncomp"]        = 0
    );
  }
  arma::mat Tused = T.cols(0, used-1);
  // alpha = Ueff * (T'T)^{-1} T' Yc
  arma::mat Ueff(n, used, arma::fill::zeros);
  for (int a = 0; a < used; ++a) Ueff.col(a) = Ucols[a];
  arma::mat G   = Tused.t() * Tused;
  arma::mat TtY = Tused.t() * Yc;
  arma::mat C;
  bool ok = arma::solve(C, G, TtY, arma::solve_opts::fast + arma::solve_opts::no_approx);
  if (!ok) ok = arma::solve(C, G, TtY);
  arma::mat alpha = Ueff * C;         // dual coefficients
  // Use bigmatrix_utils helpers for stable R objects
  Rcpp::RObject coeff_out = make_matrix_output(false, alpha.memptr(), alpha.n_rows, alpha.n_cols, "coefficients");
  Rcpp::RObject scores_out= make_matrix_output(false, Tused.memptr(), Tused.n_rows, Tused.n_cols, "scores");
  // also expose the U basis so we can compute T(new) = Kc(new,train) %*% Ueff
  Rcpp::RObject ubasis_out = make_matrix_output(false, Ueff.memptr(), Ueff.n_rows, Ueff.n_cols, "u_basis");
  return List::create(
    _["coefficients"] = coeff_out,
    // duplicate under a RKHS-friendly name used by predict()
    _["dual_coef"]    = coeff_out,
    _["intercept"]    = y_mean,
    _["scores"]       = scores_out,
    _["u_basis"]      = ubasis_out,
    _["y_means"]      = y_mean,
    _["ncomp"]        = used
  );
}


// -----------------------------------------------------------------------------
// 3) Dense RKHS-KPLS: build K with blocks, center it, then run KPLS
//   Registered as: _bigPLSR_cpp_kpls_rkhs_dense
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
SEXP cpp_kpls_rkhs_dense(Rcpp::NumericMatrix X,
                         Rcpp::RObject Y,
                         int ncomp,
                         double tol,
                         std::string kernel,
                         double gamma,
                         int degree,
                         double coef0,
                         std::string approx,
                         int approx_rank) {
  if (ncomp <= 0) stop("cpp_kpls_rkhs_dense: ncomp must be positive");
  if (X.nrow() == 0 || X.ncol() == 0) stop("cpp_kpls_rkhs_dense: empty X");
  if (Y.isNULL()) stop("cpp_kpls_rkhs_dense: Y must not be NULL");
  
  arma::mat Xa(X.begin(), X.nrow(), X.ncol(), false, true);
  const arma::uword n = Xa.n_rows;
  // Build full Gram (uncentered), compute training centering stats, then center.
  arma::mat K  = kernel_block_dense(Xa, 0, n, 0, n, kernel, gamma, degree, coef0);
  arma::rowvec k_colmeans = arma::mean(K, 0);
  double       k_mean     = arma::mean(k_colmeans);
  arma::mat Kc = center_gram_dense(K);
  // Run KPLS on centered Gram
  Rcpp::NumericMatrix Knum(n, n);
  std::copy(Kc.begin(), Kc.end(), Knum.begin());
  Rcpp::List out = cpp_kpls_from_gram(Knum, Y, ncomp, tol);
  // Attach kernel config + centering stats + algorithm tag; keep a dense training reference for predict.
  out["k_colmeans"] = k_colmeans;
  out["k_mean"]     = k_mean;
  out["kernel"]     = kernel;
  out["gamma"]      = gamma;
  out["degree"]     = degree;
  out["coef0"]      = coef0;
  out["X_ref"]      = Xa;          // used by predict.rkhs (dense path)
  out["algorithm"]  = std::string("rkhs");
  return out;
}

// -----------------------------------------------------------------------------
// // 4) Big-memory RKHS-KPLS (streaming HKH via block MVM with cpp_kernel_gram_block)
//   Registered as: _bigPLSR_cpp_kpls_rkhs_bigmem
// -----------------------------------------------------------------------------
// [[Rcpp::export]]
SEXP cpp_kpls_rkhs_bigmem(SEXP X_ptr,
                          SEXP Y,
                          int ncomp,
                          double tol,
                          std::string kernel,
                          double gamma,
                          int degree,
                          double coef0,
                          std::string approx,
                          int approx_rank,
                          int chunk_rows = 8192,
                          int chunk_cols = 8192) {
  if (TYPEOF(X_ptr) != EXTPTRSXP) stop("cpp_kpls_rkhs_bigmem: X_ptr must be an externalptr");
  Rcpp::XPtr<BigMatrix> Xbm(X_ptr);
  ensure_double_matrix(*Xbm, "X");
  const arma::uword n = Xbm->nrow();
  if (ncomp <= 0) stop("cpp_kpls_rkhs_bigmem: ncomp must be positive");
  ncomp = std::min<int>(ncomp, (int)n);
  
  // Read/center Y (stream-friendly accumulation; but we keep it simple here)
  arma::mat Ya;
  if (TYPEOF(Y) == EXTPTRSXP) {
    Rcpp::XPtr<BigMatrix> Ybm(Y);
    ensure_double_matrix(*Ybm, "Y");
    if ((arma::uword)Ybm->nrow() != n) stop("Y and X have incompatible rows");
    double* yptr = static_cast<double*>(Ybm->matrix());
    Ya = arma::mat(yptr, Ybm->nrow(), Ybm->ncol(), false, true);
  } else if (Rf_isMatrix(Y)) {
    Rcpp::NumericMatrix Ym(Y);
    if ((arma::uword)Ym.nrow() != n) stop("Y and X have incompatible rows");
    Ya = arma::mat(Ym.begin(), Ym.nrow(), Ym.ncol(), false, true);
  } else {
    Rcpp::NumericVector yv(Y);
    if ((arma::uword)yv.size() != n) stop("y and X have incompatible lengths");
    Ya = arma::mat(yv.begin(), n, 1, false, true);
  }
  arma::rowvec y_mean = arma::mean(Ya, 0);
  arma::mat    Yc     = Ya.each_row() - y_mean;
  arma::mat    Yres   = Yc;
  
  // First pass: compute training centering stats of K(X,X)
  arma::rowvec k_colmeans(n, arma::fill::zeros);
  double       k_mean_acc = 0.0;
  for (arma::uword r0 = 0; r0 < n; r0 += (arma::uword)chunk_rows) {
    arma::uword r1 = std::min<arma::uword>(n, r0 + (arma::uword)chunk_rows);
    arma::mat Kblk = kernel_block_bigmem(*Xbm, r0, r1, 0, n, kernel, gamma, degree, coef0);
    k_colmeans += arma::sum(Kblk, 0);
    k_mean_acc += arma::accu(Kblk);
  }
  k_colmeans /= static_cast<double>(n);
  double k_mean = k_mean_acc / static_cast<double>(n) / static_cast<double>(n);
  
  // Helper: multiply by HKH without building K: H K H v = H( K( H v ) )
  auto HKH_times = [&](const arma::vec& v)->arma::vec {
    arma::vec w = v - arma::mean(v);     // H v
    arma::vec u(n, arma::fill::zeros);   // u = K w (block MVM)
    for (arma::uword r0 = 0; r0 < n; r0 += (arma::uword)chunk_rows) {
      arma::uword r1 = std::min<arma::uword>(n, r0 + (arma::uword)chunk_rows);
      arma::vec acc(r1 - r0, arma::fill::zeros);
      for (arma::uword c0 = 0; c0 < n; c0 += (arma::uword)chunk_cols) {
        arma::uword c1 = std::min<arma::uword>(n, c0 + (arma::uword)chunk_cols);
        arma::mat Kblk = kernel_block_bigmem(*Xbm, r0, r1, c0, c1, kernel, gamma, degree, coef0);
        arma::vec wsub = w.rows(c0, c1 - 1);
        acc += Kblk * wsub;
      }
      u.rows(r0, r1 - 1) = acc;
    }
    u -= arma::mean(u);                   // H u
    return u;
  };
  
  arma::mat T(n, ncomp, arma::fill::zeros);
  std::vector<arma::vec> Ucols; Ucols.reserve(ncomp);
  int used = 0;
  for (int a = 0; a < ncomp; ++a) {
    arma::vec u = select_initial_u(Yres);
    arma::vec t(n, arma::fill::zeros), t_prev;
    bool ok = true;
    for (int it = 0; it < 1000; ++it) {
      arma::vec Ku = HKH_times(u);
      double tnorm = arma::norm(Ku, 2);
      if (!std::isfinite(tnorm) || tnorm <= tol) { ok = false; break; }
      t_prev = t;  t = Ku / tnorm;
      arma::vec c = Yres.t() * t / arma::dot(t,t);
      double cn2 = arma::dot(c,c);
      if (!std::isfinite(cn2) || cn2 <= tol) { ok = false; break; }
      arma::vec u_new = Yres * c / cn2;
      double rel = (t_prev.n_elem == t.n_elem && t_prev.n_elem > 0)
        ? arma::norm(t - t_prev, 2) / std::max(1.0, arma::norm(t_prev, 2))
          : arma::norm(t, 2);
      u = u_new;
      if (rel <= tol) break;
    }
    if (!ok) break;
    arma::vec c = Yres.t() * t / arma::dot(t,t);
    Yres -= t * c.t();
    T.col(a) = t;
    arma::vec Ku = HKH_times(u);  // equal to t * ||t||, but recompute safely
    double tnorm = arma::norm(Ku, 2);
    if (tnorm <= tol) break;
    Ucols.push_back(u / tnorm);   // T = (HKH) * Ueff
    used++;
  }
  if (used == 0) {
    return List::create(
      _["coefficients"] = R_NilValue,
      _["intercept"]    = y_mean,
      _["scores"]       = R_NilValue,
      _["y_means"]      = y_mean,
      _["ncomp"]        = 0
    );
  }
  
  arma::mat Tused = T.cols(0, used-1);
  // alpha = Ueff * (T'T)^{-1} T' Yc   (all small solves)
  arma::mat Ueff(n, used, arma::fill::zeros);
  for (int a = 0; a < used; ++a) Ueff.col(a) = Ucols[a];
  arma::mat G(used, used, arma::fill::zeros);
  arma::mat TtY(used, Yc.n_cols, arma::fill::zeros);
  // Compute G = T' T and T' Yc in one pass
  G  = Tused.t() * Tused;
  TtY = Tused.t() * Yc;
  arma::mat C;
  bool ok = arma::solve(C, G, TtY, arma::solve_opts::fast + arma::solve_opts::no_approx);
  if (!ok) ok = arma::solve(C, G, TtY);
  arma::mat alpha = Ueff * C;
  Rcpp::RObject coeff_out = make_matrix_output(false, alpha.memptr(), alpha.n_rows, alpha.n_cols, "coefficients");
  Rcpp::RObject scores_out= make_matrix_output(false, Tused.memptr(), Tused.n_rows, Tused.n_cols, "scores");
  // also expose the U basis so we can compute T(new) = Kc(new,train) %*% Ueff
  Rcpp::RObject ubasis_out= make_matrix_output(false, Ueff.memptr(), Ueff.n_rows, Ueff.n_cols, "u_basis");
  Rcpp::List out = List::create(
    _["coefficients"] = coeff_out,  // dual alpha (n x m)
    // duplicate under a RKHS-friendly name used by predict()
    _["dual_coef"]    = coeff_out,
    _["intercept"]    = y_mean,
    _["scores"]       = scores_out,
    _["u_basis"]      = ubasis_out,
    _["y_means"]      = y_mean,
    _["ncomp"]        = used
  );
  // Attach kernel config + centering stats + algorithm tag.
  out["k_colmeans"] = k_colmeans;
  out["k_mean"]     = k_mean;
  out["kernel"]     = kernel;
  out["gamma"]      = gamma;
  out["degree"]     = degree;
  out["coef0"]      = coef0;
  out["algorithm"]  = std::string("rkhs");
  return out;
}

