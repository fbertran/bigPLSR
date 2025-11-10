#include <RcppArmadillo.h>
using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

// ---------- Minimal SIMPLS-from-cross helper (pure C++) --------------------
// If you already have a shared helper, replace this with your header include.
static Rcpp::List simpls_from_cross_arma(const arma::mat& XtX,
                                         const arma::mat& XtY,
                                         const arma::rowvec& x_means,
                                         const arma::rowvec& y_means,
                                         int ncomp,
                                         double tol)
{
  const arma::uword p = XtX.n_rows;
  const arma::uword m = XtY.n_cols;
  ncomp = std::max(0, std::min<int>(ncomp, std::min<arma::uword>(p, XtX.n_cols)));
  
  arma::mat W(p, ncomp, arma::fill::zeros);
  arma::mat P(p, ncomp, arma::fill::zeros);
  arma::mat Q(m, ncomp, arma::fill::zeros);
  
  arma::mat V = arma::eye<arma::mat>(p, p);           // deflation projector
  arma::mat S = XtY;                                   // p x m   (C_xy)
  arma::mat C = XtX;                                   // p x p   (C_xx)
  
  int used = 0;
  for (int a = 0; a < ncomp; ++a) {
    // dominant vector of S S^T under C metric -> eig of C^{-1} S S^T
    arma::mat Cinvt;
    arma::mat Csym = 0.5 * (C + C.t());
    bool ok = arma::inv_sympd(Cinvt, Csym);
    if (!ok) { // damp if needed
      arma::mat Creg = Csym + 1e-10 * arma::eye<arma::mat>(p, p);
      ok = arma::inv_sympd(Cinvt, Creg);
    }
    if (!ok) break;
    
    arma::mat A = Cinvt * (S * S.t());                 // p x p
    A = 0.5 * (A + A.t());                             // enforce symmetry
    arma::vec eigval; arma::mat eigvec;
    ok = arma::eig_sym(eigval, eigvec, A);
    if (!ok || eigval.n_elem == 0) break;
    arma::uword idx = eigval.n_elem - 1;
    arma::vec w = eigvec.col(idx);
    double wn = arma::norm(w, 2.0);
    if (!std::isfinite(wn) || wn <= tol) break;
    w /= wn;
    
    arma::vec t = (XtX * w);                           // proxy for scores direction
    double tn = arma::norm(t, 2.0);
    if (!std::isfinite(tn) || tn <= tol) break;
    t /= tn;
    
    arma::vec pvec = XtX * w;                          // p x 1
    double denom = arma::as_scalar(w.t() * pvec);
    if (!std::isfinite(denom) || std::fabs(denom) <= tol) break;
    
    arma::rowvec qvec = (w.t() * XtY) / denom;         // 1 x m
    
    W.col(a) = w;
    P.col(a) = pvec / denom;                           // standard scaling
    Q.col(a) = qvec.t();
    // Deflation in X-space:
    arma::mat Pw = P.col(a) * W.col(a).t();            // p x p
    V = V - Pw;
    // Update moments under deflation:
    C = V.t() * XtX * V;
    C = 0.5 * (C + C.t());
    S = V.t() * XtY;
    
    used++;
  }
  
  if (used == 0) {
    return List::create(
      _["coefficients"] = NumericMatrix(p, m),
      _["intercept"]    = y_means,
      _["x_weights"]    = R_NilValue,
      _["x_loadings"]   = R_NilValue,
      _["y_loadings"]   = R_NilValue,
      _["scores"]       = R_NilValue,
      _["x_means"]      = x_means,
      _["y_means"]      = y_means,
      _["ncomp"]        = 0
    );
  }
  
  W  = W.cols(0, used-1);
  P  = P.cols(0, used-1);
  Q  = Q.cols(0, used-1);
  
  arma::mat R = P.t() * W;                             // A x A
  R = 0.5 * (R + R.t());
  arma::mat Rinv; arma::inv(Rinv, R);
  arma::mat beta = W * Rinv * Q.t();                   // p x m
  arma::rowvec intercept = y_means - x_means * beta;   // 1 x m
  
  return List::create(
    _["coefficients"] = beta,
    _["intercept"]    = intercept,
    _["x_weights"]    = W,
    _["x_loadings"]   = P,
    _["y_loadings"]   = Q,
    _["scores"]       = R_NilValue, // compute later if needed
    _["x_means"]      = x_means,
    _["y_means"]      = y_means,
    _["ncomp"]        = used
  );
}

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
  
  // Build batch SIMPLS from the current EWMA moments
  Rcpp::List fit = simpls_from_cross_arma(S.Cxx, S.Cxy, S.mu_x, S.mu_y, S.A, tol);
  
  // Tag algorithm to integrate with your R predict() flow
  fit.push_back(Rcpp::String("kf_pls"), "algorithm");
  
  return fit;
}