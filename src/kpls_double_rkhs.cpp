#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

using namespace Rcpp;

// [[Rcpp::plugins(cpp17)]]

[[maybe_unused]] static inline void ensure_double_bigmatrix(const BigMatrix& M, const char* nm){
  if (M.matrix_type() != 8) stop(std::string(nm) + " must be a double big.matrix");
}

// ---------- small kernel helpers (dense) ----------
static inline arma::mat k_linear(const arma::mat& A, const arma::mat& B) {
  return A * B.t();
}
static inline arma::mat k_poly(const arma::mat& A, const arma::mat& B,
                               double gamma, int degree, double c0) {
  arma::mat G = gamma * (A * B.t()) + c0;
  arma::mat K(G.n_rows, G.n_cols, arma::fill::ones);
  for (int d = 0; d < degree; ++d) K %= G;
  return K;
}
static inline arma::mat k_rbf(const arma::mat& A, const arma::mat& B, double gamma) {
  arma::vec an = arma::sum(arma::square(A), 1);
  arma::vec bn = arma::sum(arma::square(B), 1);
  arma::mat G  = A * B.t();
  arma::mat D  = arma::repmat(an,1,B.n_rows) + arma::repmat(bn.t(),A.n_rows,1) - 2.0*G;
  return arma::exp(-gamma * D);
}
static inline arma::mat k_tanh(const arma::mat& A, const arma::mat& B,
                               double gamma, double c0) {
  arma::mat G = gamma * (A * B.t()) + c0;
  return arma::tanh(G);
}
static inline arma::mat make_kernel(const arma::mat& A, const arma::mat& B,
                                    const std::string& kernel,
                                    double gamma, int degree, double c0) {
  if (kernel == "linear") return k_linear(A,B);
  if (kernel == "poly"   || kernel == "polynomial") return k_poly(A,B,gamma,degree,c0);
  if (kernel == "rbf"    || kernel == "gaussian")   return k_rbf(A,B,gamma);
  if (kernel == "tanh")  return k_tanh(A,B,gamma,c0);
  stop("rkhs_xy: unknown kernel");
}

// Double-centering for Gram matrices: Kc = H K H
static inline arma::mat center_gram(arma::mat K) {
  arma::rowvec c = arma::mean(K, 0);
  arma::colvec r = arma::mean(K, 1);
  double mu = arma::mean(c);
  K.each_row() -= c;
  K.each_col() -= r;
  K += mu;
  return K;
}

// Apply M(v) = Lx^{-1} * ( Kx * ( Ky * ( Kx * ( Lx^{-1} v ) ) ) )
// where Lx = Kx + lambda_x I.
// This is a symmetric, PSD “Kx^{1/2} Ky Kx^{1/2}”-style operator without explicit sqrt.
static inline arma::vec apply_M(const arma::mat& Kx, const arma::mat& Lx,
                                const arma::mat& Ky, const arma::vec& v) {
  arma::vec z1 = arma::solve(Lx, v, arma::solve_opts::fast);   // Lx^{-1} v
  arma::vec a1 = Kx * z1;
  arma::vec a2 = Ky * a1;
  arma::vec a3 = Kx * a2;
  arma::vec out= arma::solve(Lx, a3, arma::solve_opts::fast);  // Lx^{-1} (Kx Ky Kx Lx^{-1} v)
  return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_kpls_rkhs_xy_dense(Rcpp::NumericMatrix X_,
                                  Rcpp::RObject Y_,
                                  int ncomp,
                                  double tol,
                                  std::string kernel_x,
                                  double gamma_x,
                                  int degree_x,
                                  double coef0_x,
                                  std::string kernel_y,
                                  double gamma_y,
                                  int degree_y,
                                  double coef0_y,
                                  double lambda_x,
                                  double lambda_y) {
  if (ncomp <= 0) stop("rkhs_xy: ncomp must be positive");
  arma::mat X(X_.begin(), X_.nrow(), X_.ncol(), false, true);
  arma::mat Y;
  if (Rf_isMatrix(Y_)) {
    Rcpp::NumericMatrix Ym(Y_);
    if (Ym.nrow() != X.n_rows) stop("rkhs_xy: X and Y must have same number of rows");
    Y = arma::mat(Ym.begin(), Ym.nrow(), Ym.ncol(), false, true);
  } else {
    Rcpp::NumericVector yv(Y_);
    if ((int)yv.size() != X.n_rows) stop("rkhs_xy: y length must equal nrow(X)");
    Y = arma::mat(yv.begin(), X.n_rows, 1, false, true);
  }
  const arma::uword n = X.n_rows;
  ncomp = std::min<int>(ncomp, (int)n);
  
  // Build and center Gram matrices
  arma::mat Kx = make_kernel(X, X, kernel_x, gamma_x, degree_x, coef0_x);
  arma::mat Ky;
  // Build Y Gram from samples (rows)
  Ky = make_kernel(Y, Y, kernel_y, gamma_y, degree_y, coef0_y);
  Kx = center_gram(Kx);
  Ky = center_gram(Ky);
  
  // Ridges (for stability)
  if (lambda_x > 0) Kx.diag() += lambda_x;
  if (lambda_y > 0) Ky.diag() += lambda_y;
  arma::mat Lx = Kx; // Lx = Kx (+ lambda_x I) already applied
  
  // Power iterations to get T = [t1..tA] (orthonormal in Euclidean metric)
  arma::mat T(n, ncomp, arma::fill::zeros);
  int used = 0;
  arma::vec v = arma::randn<arma::vec>(n); // random start
  for (int a = 0; a < ncomp; ++a) {
    // re-start from previous tail to help convergence
    if (a > 0) v = arma::randn<arma::vec>(n);
    // power iteration with Gram-Schmidt against previous T
    for (int it = 0; it < 800; ++it) {
      arma::vec w = apply_M(Kx, Lx, Ky, v);
      // orthogonalize vs previous components
      for (int j = 0; j < used; ++j) {
        double proj = arma::dot(T.col(j), w);
        w -= T.col(j) * proj;
      }
      double wn = arma::norm(w, 2);
      if (!std::isfinite(wn) || wn <= tol) break;
      arma::vec v_new = w / wn;
      double diff = arma::norm(v_new - v, 2);
      v = v_new;
      if (diff <= tol) break;
    }
    double vn = arma::norm(v, 2);
    if (!std::isfinite(vn) || vn <= tol) break;
    T.col(used) = v / vn;
    used++;
  }
  if (used == 0) {
    arma::rowvec y_mean = arma::mean(Y, 0);
    return List::create(
      _["coefficients"] = R_NilValue,
      _["dual_coef"]    = R_NilValue,
      _["scores"]       = R_NilValue,
      _["intercept"]    = y_mean,
      _["y_means"]      = y_mean,
      _["ncomp"]        = 0
    );
  }
  arma::mat Tused = T.cols(0, used-1);
  
  // Map T into Kx-dual space so that Kx * U ≈ T  => U = Lx^{-1} T (since Lx = Kx + λ_x I)
  arma::mat U = arma::solve(Lx, Tused, arma::solve_opts::fast);
  
  // Center Y and compute small regression in latent space:
  arma::rowvec y_mean = arma::mean(Y, 0);
  arma::mat Yc = Y.each_row() - y_mean;
  arma::mat G  = Tused.t() * Tused;               // used x used
  arma::mat TtY= Tused.t() * Yc;                  // used x m
  arma::mat C;
  bool ok = arma::solve(C, G, TtY, arma::solve_opts::fast + arma::solve_opts::no_approx);
  if (!ok) ok = arma::solve(C, G, TtY);
  
  // Dual coefficients α so that Ŷ ≈ Kx_c * α + 1*y_mean
  arma::mat alpha = U * C;
  
  return List::create(
    _["coefficients"] = alpha,
    _["dual_coef"]    = alpha,
    _["scores"]       = Tused,
    _["intercept"]    = y_mean,
    _["y_means"]      = y_mean,
    _["ncomp"]        = used
  );
}


