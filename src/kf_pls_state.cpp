#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

// ---------- KF-PLS minimal state -------------------------------------------
struct KFPLSState {
  int p{0}, m{0}, A{0};
  double lambda{0.99};       // forgetting factor
  double q_proc{0.0};        // process noise magnitude (ridge on Cxx)
  double r_meas{0.0};        // reserved (not used in this minimal API)
  
  bool initialized{false};
  
  arma::rowvec mu_x;         // 1 x p  (EWMA)
  arma::rowvec mu_y;         // 1 x m
  arma::mat Cxx;             // p x p
  arma::mat Cxy;             // p x m
  
  KFPLSState(int p_, int m_, int A_, double lambda_, double q_, double r_)
    : p(p_), m(m_), A(A_), lambda(lambda_), q_proc(q_), r_meas(r_),
      mu_x(arma::rowvec(p_, arma::fill::zeros)),
      mu_y(arma::rowvec(m_, arma::fill::zeros)),
      Cxx(arma::mat(p_, p_, arma::fill::zeros)),
      Cxy(arma::mat(p_, m_, arma::fill::zeros)) {}
};

[[maybe_unused]] static void kf_state_finalizer(KFPLSState* ptr) {
  if (ptr) delete ptr;
}

// ---------- Rcpp API --------------------------------------------------------

// [[Rcpp::export]]
SEXP cpp_kf_pls_state_new(int p, int m, int ncomp,
                          double lambda = 0.99,
                          double q_proc = 0.0,
                          double r_meas = 0.0)
{
  if (p <= 0 || m <= 0 || ncomp <= 0) stop("p, m, ncomp must be positive");
  KFPLSState* st = new KFPLSState(p, m, ncomp, lambda, q_proc, r_meas);
  Rcpp::XPtr<KFPLSState> xp(st, true);
  return xp;
}

// [[Rcpp::export]]
void cpp_kf_pls_state_update(SEXP state_xptr,
                             Rcpp::NumericMatrix X_,
                             Rcpp::RObject Y_)
{
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
    Y = arma::mat(yv.begin(), X.n_rows, 1, false, true);
    if (S.m != 1) stop("State expects m columns in Y");
  }
  
  // EWMA of means: use alpha_mean = (1 - lambda)
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
  
  // Forgetting + process noise on Cxx, forgetting on Cxy
  S.Cxx = S.lambda * S.Cxx + Xc.t() * Xc;
  if (S.q_proc > 0.0) S.Cxx += S.q_proc * arma::eye<arma::mat>(S.p, S.p);
  S.Cxx = 0.5 * (S.Cxx + S.Cxx.t());
  S.Cxy = S.lambda * S.Cxy + Xc.t() * Yc;
  
  // (Optional branch (ii): Kalman update on a parameter vector theta with R)
  // Minimal API omits this to keep updates O(p^2 + p m).
}

// [[Rcpp::export]]
Rcpp::List cpp_kf_pls_state_fit(SEXP state_xptr,
                                double tol = 1e-8)
{
  Rcpp::XPtr<KFPLSState> xp(state_xptr);
  KFPLSState& S = *xp;
  
  arma::mat Cxx_sym = 0.5 * (S.Cxx + S.Cxx.t());
  
  Rcpp::NumericMatrix XtX(Cxx_sym.n_rows, Cxx_sym.n_cols);
  std::copy(Cxx_sym.begin(), Cxx_sym.end(), XtX.begin());
  
  Rcpp::NumericMatrix XtY(S.Cxy.n_rows, S.Cxy.n_cols);
  std::copy(S.Cxy.begin(), S.Cxy.end(), XtY.begin());
  
  Rcpp::NumericVector xmean(S.mu_x.begin(), S.mu_x.end());
  Rcpp::NumericVector ymean(S.mu_y.begin(), S.mu_y.end());
  
  Rcpp::Environment ns = Rcpp::Environment::namespace_env("bigPLSR");
  Rcpp::Function simpls = ns["cpp_simpls_from_cross"];
  
  Rcpp::List fit = simpls(XtX, XtY, xmean, ymean,
                          Rcpp::wrap(S.A), Rcpp::wrap(tol));
  
  // Tag algorithm to integrate with your R predict() flow
  fit.push_back(Rcpp::String("kf_pls"), "algorithm");
  
  return fit;
}