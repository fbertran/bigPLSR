#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// Logistic sigmoid with clamping to avoid 0/1 exactly
static inline arma::vec sigmoid(const arma::vec& x) {
  arma::vec y = 1.0 / (1.0 + arma::exp(-x));
  // clamp
  y = arma::clamp(y, 1e-12, 1.0 - 1e-12);
  return y;
}

//' Fast IRLS for binomial logit with class weights
//'
//' @param T n x A numeric matrix of latent scores (no intercept column)
//' @param ybin integer vector of {0,1} labels (length n)
//' @param w_class optional length-2 numeric vector: weights for classes c( w0, w1 )
//' @param maxit max IRLS iterations
//' @param tol relative tolerance on parameter change
//'
//' @return list(beta = A-vector, b = scalar intercept, fitted = n-vector,
//'              iter = integer, converged = logical)
//'
// [[Rcpp::export]]
Rcpp::List cpp_irls_binomial(const arma::mat& T,
                              const Rcpp::IntegerVector& ybin,
                              Rcpp::Nullable<Rcpp::NumericVector> w_class = R_NilValue,
                              int maxit = 50,
                              double tol = 1e-8) {
   const arma::uword n = T.n_rows;
   const arma::uword A = T.n_cols;
   if ((arma::uword)ybin.size() != n) Rcpp::stop("cpp_irls_binomial: y length must match nrow(T)");
   
   arma::vec y(n);
   for (arma::uword i = 0; i < n; ++i) {
     int yi = ybin[i];
     if (yi != 0 && yi != 1) Rcpp::stop("cpp_irls_binomial: y must be 0/1");
     y[i] = (double)yi;
   }
   
   // class weights (w0 for y=0, w1 for y=1)
   double w0 = 1.0, w1 = 1.0;
   if (w_class.isNotNull()) {
     Rcpp::NumericVector cw(w_class);
     if (cw.size() >= 2) {
       w0 = cw[0];
       w1 = cw[1];
     } else if (cw.size() == 1) {
       w1 = cw[0];
     }
   }
   
   // Design matrix X = [1, T]
   arma::mat X(n, A + 1, arma::fill::ones);
   if (A > 0) X.cols(1, A) = T;
   
   // Params: theta = [b; beta]
   arma::vec theta(A + 1, arma::fill::zeros);
   
   bool converged = false;
   int it = 0;
   
   // Small ridge in case of near-singular normal equations
   const double ridge = 1e-10;
   
   for (it = 0; it < maxit; ++it) {
     arma::vec eta = X * theta;             // n
     arma::vec p   = sigmoid(eta);          // n
     
     // weights: w_i = class_weight_i * p_i (1 - p_i)
     arma::vec w(n);
     for (arma::uword i = 0; i < n; ++i) {
       double cw_i = (y[i] > 0.5) ? w1 : w0;
       w[i] = cw_i * (p[i] * (1.0 - p[i]));
       if (w[i] < 1e-16) w[i] = 1e-16;
     }
     
     // working response: z = eta + (y - p)/[p(1-p)]
     arma::vec z = eta + (y - p) / arma::clamp(p % (1.0 - p), 1e-12, 1.0);
     
     // Solve (X' W X + ridge I) theta = X' W z
     arma::mat WX = X.each_col() % w;       // n x (A+1)
     arma::mat XtWX = X.t() * WX;           // (A+1) x (A+1)
     XtWX.diag() += ridge;
     arma::vec XtWz = X.t() * (w % z);
     
     arma::vec theta_new;
     bool ok = arma::solve(theta_new, XtWX, XtWz, arma::solve_opts::fast + arma::solve_opts::no_approx);
     if (!ok) ok = arma::solve(theta_new, XtWX, XtWz);
     if (!ok) {
       // as last resort, use pinv
       theta_new = arma::pinv(XtWX) * XtWz;
     }
     
     double num = arma::norm(theta_new - theta, 2);
     double den = std::max(1.0, arma::norm(theta, 2));
     theta = theta_new;
     if (num / den <= tol) { converged = true; break; }
   }
   
   arma::vec eta = X * theta;
   arma::vec p   = sigmoid(eta);
   
   double b = theta[0];
   arma::vec beta = (A > 0) ? theta.subvec(1, A) : arma::vec();
   
   return Rcpp::List::create(
     Rcpp::Named("beta")      = beta,
     Rcpp::Named("b")         = b,
     Rcpp::Named("fitted")    = p,
     Rcpp::Named("iter")      = it + 1,
     Rcpp::Named("converged") = converged
   );
 }
