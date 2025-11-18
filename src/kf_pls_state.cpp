// kf_pls_state.cpp
#include <RcppArmadillo.h>
#include <Rinternals.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

struct KFPLSState {
  int p{0}, m{0}, A{0};
  double lambda{0.99};   // forgetting factor (close to 1 = slow decay)
  double q_proc{0.0};    // process noise (ridge on Cxx)
  double r_meas{0.0};    // reserved for full KF (not used here)
  
  bool initialized{false};
  bool exact_mode{false};     // exact accumulation mode if lambda≈1 && q_proc==0
  
  arma::rowvec mu_x;     // 1×p EWMA mean of X
  arma::rowvec mu_y;     // 1×m EWMA mean of Y
  arma::mat    Cxx;      // p×p EWMA cross-product (cov scale up to constant)
  arma::mat    Cxy;      // p×m EWMA cross-product
  
  // exact accumulators (used only if exact_mode == true)
  double       N_acc{0.0};
  arma::rowvec SX;       // 1×p sum of X rows
  arma::rowvec SY;       // 1×m sum of Y rows
  arma::mat    SXX;      // p×p sum of X'X over all chunks
  arma::mat    SXY;      // p×m sum of X'Y over all chunks
    
  KFPLSState(int p_, int m_, int A_, double lambda_, double q_, double r_)
    : p(p_), m(m_), A(A_), lambda(lambda_), q_proc(q_), r_meas(r_),
      mu_x(p_, arma::fill::zeros),
      mu_y(m_, arma::fill::zeros),
      Cxx(p_, p_, arma::fill::zeros),
      Cxy(p_, m_, arma::fill::zeros)
  {
    // Enter exact (batch-parity) mode only when truly asked for:
    // - lambda == 1 (within tight epsilon), and
    // - no process noise
    const double eps = 1e-12;
    exact_mode = (std::abs(lambda_ - 1.0) <= eps && q_ == 0.0);
    if (exact_mode) {
      N_acc = 0.0;
      SX.set_size(1, p_); SX.zeros();
      SY.set_size(1, m_); SY.zeros();
      SXX.set_size(p_, p_); SXX.zeros();
      SXY.set_size(p_, m_); SXY.zeros();
    }
  }
};

// --- R registered C symbol from cpp_simpls_from_cross -----------------------
// Signature: SEXP _bigPLSR_cpp_simpls_from_cr​oss(SEXP XtX, SEXP XtY, SEXP xmean, SEXP ymean, SEXP ncomp, SEXP tol);
extern "C" SEXP _bigPLSR_cpp_simpls_from_cross(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

// [[Rcpp::export]]
SEXP cpp_kf_pls_state_new(int p, int m, int ncomp,
                          double lambda = 0.99,
                          double q_proc = 0.0,
                          double r_meas = 0.0) {
  if (p <= 0 || m <= 0 || ncomp <= 0) stop("p, m, ncomp must be positive");
  auto* st = new KFPLSState(p, m, ncomp, lambda, q_proc, r_meas);
  // use default delete finalizer managed by XPtr
  Rcpp::XPtr<KFPLSState> xp(st, true);
  return xp;
}

// [[Rcpp::export]]
void cpp_kf_pls_state_update(SEXP state_xptr,
                             Rcpp::NumericMatrix X_,
                             Rcpp::RObject Y_) {
  Rcpp::XPtr<KFPLSState> xp(state_xptr);
  KFPLSState& S = *xp;
  
  arma::mat X(X_.begin(), X_.nrow(), X_.ncol(), false, true);
  if ((int)X.n_cols != S.p) stop("X chunk has wrong number of columns");
  
  arma::mat Y;
  if (Rf_isMatrix(Y_)) {
    Rcpp::NumericMatrix Ym(Y_);
    if ((int)Ym.nrow() != (int)X.n_rows || (int)Ym.ncol() != S.m)
      stop("Y chunk has incompatible dimensions");
    Y = arma::mat(Ym.begin(), Ym.nrow(), Ym.ncol(), false, true);
  } else {
    Rcpp::NumericVector yv(Y_);
    if ((int)yv.size() != (int)X.n_rows)
      stop("y chunk length must match X rows");
    if (S.m != 1) stop("State expects Y with m columns");
    Y = arma::mat(yv.begin(), X.n_rows, 1, false, true);
  }

  if (S.exact_mode) {
    // Exact sufficient statistics (no EWMA means here)
    S.N_acc += static_cast<double>(X.n_rows);
    S.SX    += arma::sum(X, 0);
    S.SY    += arma::sum(Y, 0);
    S.SXX   += X.t() * X;
    S.SXY   += X.t() * Y;
  } else {
    // EWMA means & centered cross-moments
    double alpha_mean = 1.0 - S.lambda;
    arma::rowvec mx = arma::mean(X, 0);
    arma::rowvec my = arma::mean(Y, 0);
    if (!S.initialized) {
      S.mu_x = mx;
      S.mu_y = my;
      S.initialized = true;
    } else {
      S.mu_x = (1.0 - alpha_mean) * S.mu_x + alpha_mean * mx;
      S.mu_y = (1.0 - alpha_mean) * S.mu_y + alpha_mean * my;
    }
    arma::mat Xc = X.each_row() - S.mu_x;
    arma::mat Yc = Y.each_row() - S.mu_y;
    S.Cxx = S.lambda * S.Cxx + Xc.t() * Xc;
    if (S.q_proc > 0.0) S.Cxx += S.q_proc * arma::eye<arma::mat>(S.p, S.p);
    S.Cxx = 0.5 * (S.Cxx + S.Cxx.t());
    S.Cxy = S.lambda * S.Cxy + Xc.t() * Yc;
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_kf_pls_state_fit(SEXP state_xptr, double tol = 1e-8) {
  Rcpp::XPtr<KFPLSState> xp(state_xptr);
  KFPLSState& S = *xp;
  
  Rcpp::NumericMatrix XtX(S.p, S.p);
  Rcpp::NumericMatrix XtY(S.p, S.m);
  Rcpp::NumericVector xmean(S.p), ymean(S.m);
  
  if (S.exact_mode) {
    // global means (from sums) and centered cross-products
    const double N = (S.N_acc > 0.0) ? S.N_acc : 1.0;
    arma::rowvec mx = S.SX / N;
    arma::rowvec my = S.SY / N;
    // Centered cross-products (equal to crossprod(Xc), crossprod(Xc, Yc))
    // using numerically stable formula: SX' SX / N = N * (mx' mx).
    arma::mat Cxx = S.SXX - (S.SX.t() * S.SX) / N;
    arma::mat Cxy = S.SXY - (S.SX.t() * S.SY) / N;
    Cxx = 0.5 * (Cxx + Cxx.t());
    std::copy(Cxx.begin(), Cxx.end(), XtX.begin());
    std::copy(Cxy.begin(), Cxy.end(), XtY.begin());
    std::copy(mx.begin(), mx.end(), xmean.begin());
    std::copy(my.begin(), my.end(), ymean.begin());
  } else {
    arma::mat Cxx_sym = 0.5 * (S.Cxx + S.Cxx.t());
    std::copy(Cxx_sym.begin(), Cxx_sym.end(), XtX.begin());
    std::copy(S.Cxy.begin(), S.Cxy.end(), XtY.begin());
    std::copy(S.mu_x.begin(), S.mu_x.end(), xmean.begin());
    std::copy(S.mu_y.begin(), S.mu_y.end(), ymean.begin());
  }
  
  SEXP fitSEXP = _bigPLSR_cpp_simpls_from_cross(
    XtX, XtY, xmean, ymean, Rcpp::wrap(S.A), Rcpp::wrap(tol));
  Rcpp::List fit(fitSEXP);
  fit["algorithm"] = Rcpp::String("kf_pls");
  return fit;
}