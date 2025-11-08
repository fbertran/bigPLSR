#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include <cmath>

//' Internal kernel and wide-kernel PLS solver
//'
//' @param X Centered design matrix.
//' @param Y Centered response matrix.
//' @param ncomp Maximum number of components.
//' @param tol Numerical tolerance.
//' @param wide Whether to use the wide-kernel update.
//' @return A list containing the kernel PLS factors.
//'
// [[Rcpp::export]]
Rcpp::List cpp_kernel_pls(const arma::mat& X,
                           const arma::mat& Y,
                           int ncomp,
                           double tol,
                           bool wide) {
   const arma::uword n = X.n_rows;
   const arma::uword p = X.n_cols;
   const arma::uword m = Y.n_cols;
   
   if (n == 0 || p == 0 || m == 0) {
     Rcpp::stop("Invalid data dimensions for kernel PLS");
   }
   if (ncomp <= 0) {
     Rcpp::stop("`ncomp` must be positive");
   }
   
   arma::vec x_means = arma::mean(X, 0).t();
   arma::vec y_means = arma::mean(Y, 0).t();
   arma::mat Xc = X;
   arma::mat Yc = Y;
   Xc.each_row() -= x_means.t();
   Yc.each_row() -= y_means.t();
   
   const arma::uword max_comp = std::min({static_cast<arma::uword>(ncomp), n, p});
   if (max_comp == 0) {
     return Rcpp::List::create(
       Rcpp::Named("coefficients") = arma::mat(p, m, arma::fill::zeros),
       Rcpp::Named("intercept") = y_means,
       Rcpp::Named("x_weights") = arma::mat(p, 0, arma::fill::zeros),
       Rcpp::Named("x_loadings") = arma::mat(p, 0, arma::fill::zeros),
       Rcpp::Named("y_loadings") = arma::mat(m, 0, arma::fill::zeros),
       Rcpp::Named("scores") = arma::mat(n, 0, arma::fill::zeros),
       Rcpp::Named("x_means") = x_means,
       Rcpp::Named("y_means") = y_means,
       Rcpp::Named("ncomp") = 0
     );
   }
   
   arma::mat W(p, max_comp, arma::fill::zeros);
   arma::mat P(p, max_comp, arma::fill::zeros);
   arma::mat Q(m, max_comp, arma::fill::zeros);
   arma::mat Tmat(n, max_comp, arma::fill::zeros);
   
   arma::mat X_res = Xc;
   arma::mat Y_res = Yc;
   
   arma::uword actual = 0;
   for (arma::uword h = 0; h < max_comp; ++h) {
     arma::vec u = Y_res.col(0);
     u.replace(arma::datum::nan, 0.0);
     double u_norm = arma::norm(u, 2);
     if (!std::isfinite(u_norm) || u_norm <= tol) {
       u = arma::randn<arma::vec>(n);
       u_norm = arma::norm(u, 2);
       if (!std::isfinite(u_norm) || u_norm <= tol) {
         break;
       }
     }
     u /= u_norm;
     
     for (int iter = 0; iter < 50; ++iter) {
       arma::vec inner = wide ? (X_res.t() * u) : (X_res.t() * u);
       arma::vec t_vec = X_res * inner;
       double t_norm = arma::norm(t_vec, 2);
       if (!std::isfinite(t_norm) || t_norm <= tol) {
         break;
       }
       t_vec /= t_norm;
       arma::vec c_vec = Y_res.t() * t_vec;
       if (arma::all(arma::abs(c_vec) <= tol)) {
         break;
       }
       arma::vec u_new = Y_res * c_vec;
       double u_new_norm = arma::norm(u_new, 2);
       if (!std::isfinite(u_new_norm) || u_new_norm <= tol) {
         break;
       }
       u_new /= u_new_norm;
       if (arma::norm(u_new - u, 2) <= tol) {
         u = u_new;
         break;
       }
       u = u_new;
     }
     
     arma::vec inner = wide ? (X_res.t() * u) : (X_res.t() * u);
     arma::vec t_vec = X_res * inner;
     double t_norm = arma::norm(t_vec, 2);
     if (!std::isfinite(t_norm) || t_norm <= tol) {
       break;
     }
     t_vec /= t_norm;
     
     double denom = arma::dot(t_vec, t_vec);
     if (!std::isfinite(denom) || denom <= tol) {
       break;
     }
     
     arma::vec p_vec = X_res.t() * t_vec / denom;
     arma::vec q_vec = Y_res.t() * t_vec / denom;
     arma::vec w_vec = Xc.t() * t_vec;
     double w_norm = arma::norm(w_vec, 2);
     if (std::isfinite(w_norm) && w_norm > tol) {
       w_vec /= w_norm;
     }
     
     X_res -= t_vec * p_vec.t();
     Y_res -= t_vec * q_vec.t();
     
     W.col(actual) = w_vec;
     P.col(actual) = p_vec;
     Q.col(actual) = q_vec;
     Tmat.col(actual) = t_vec;
     ++actual;
     
     if (X_res.n_rows == 0 || X_res.n_cols == 0) {
       break;
     }
   }
   
   if (actual == 0) {
     return Rcpp::List::create(
       Rcpp::Named("coefficients") = arma::mat(p, m, arma::fill::zeros),
       Rcpp::Named("intercept") = y_means,
       Rcpp::Named("x_weights") = arma::mat(p, 0, arma::fill::zeros),
       Rcpp::Named("x_loadings") = arma::mat(p, 0, arma::fill::zeros),
       Rcpp::Named("y_loadings") = arma::mat(m, 0, arma::fill::zeros),
       Rcpp::Named("scores") = arma::mat(n, 0, arma::fill::zeros),
       Rcpp::Named("x_means") = x_means,
       Rcpp::Named("y_means") = y_means,
       Rcpp::Named("ncomp") = 0
     );
   }
   
   arma::mat W_sub = W.cols(0, actual - 1);
   arma::mat P_sub = P.cols(0, actual - 1);
   arma::mat Q_sub = Q.cols(0, actual - 1);
   arma::mat T_sub = Tmat.cols(0, actual - 1);
   
   arma::mat Rmat = P_sub.t() * W_sub;
   arma::mat Rinv;
   bool ok = arma::inv(Rinv, Rmat);
   arma::mat beta(p, m, arma::fill::zeros);
   if (ok) {
     beta = W_sub * Rinv * Q_sub.t();
   }
   
   arma::vec intercept = y_means - beta.t() * x_means;
   
   return Rcpp::List::create(
     Rcpp::Named("coefficients") = beta,
     Rcpp::Named("intercept") = intercept,
     Rcpp::Named("x_weights") = W_sub,
     Rcpp::Named("x_loadings") = P_sub,
     Rcpp::Named("y_loadings") = Q_sub,
     Rcpp::Named("scores") = T_sub,
     Rcpp::Named("x_means") = x_means,
     Rcpp::Named("y_means") = y_means,
     Rcpp::Named("ncomp") = static_cast<int>(actual)
   );
 }
 