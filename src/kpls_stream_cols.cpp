#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
using namespace Rcpp;

// [[Rcpp::export]]
Rcpp::List cpp_kpls_stream_cols(SEXP X_ptr, SEXP Y_ptr,
                                int ncomp, int chunk_cols,
                                bool center,
                                bool return_big) {
  Rcpp::XPtr<BigMatrix> Xp(X_ptr);
  Rcpp::XPtr<BigMatrix> Yp(Y_ptr);
  const std::size_t n = Xp->nrow();
  const std::size_t p = Xp->ncol();
  const std::size_t m = Yp->ncol();
  arma::mat X(static_cast<double*>(Xp->matrix()), n, p, false, true);
  arma::mat Y(static_cast<double*>(Yp->matrix()), n, m, false, true);
  arma::rowvec mx = arma::mean(X, 0);
  arma::rowvec my = arma::mean(Y, 0);
  if (center) { X.each_row() -= mx; Y.each_row() -= my; }
  const int H = std::min<int>(ncomp, std::min<std::size_t>(n, p));
  arma::mat W, P, Q, T;
  W.set_size(p, H); P.set_size(p, H); Q.set_size(m, H); T.set_size(n, H);
  arma::mat Xres = X, Yres = Y;
  arma::vec u = Yres.col(0);
  int used = 0;
  for (int h = 0; h < H; ++h) {
    if (arma::norm(u,2) <= 1e-14) break;
    arma::vec a = Xres.t() * u; a /= arma::norm(a,2);
    arma::vec t = Xres * a;
    double t2 = arma::dot(t,t); if (!(t2>0)) break;
    arma::vec pvec = Xres.t() * t / t2;
    arma::vec qvec = Yres.t() * t / t2;
    Xres -= t * pvec.t();
    Yres -= t * qvec.t();
    W.col(h)=a; P.col(h)=pvec; Q.col(h)=qvec; T.col(h)=t; ++used;
    u = Yres * qvec;
  }
  if (used==0) return List::create(_["coefficients"]=R_NilValue);
  W = W.cols(0,used-1); P=P.cols(0,used-1); Q=Q.cols(0,used-1); T=T.cols(0,used-1);
  arma::mat R = P.t()*W, Rinv; arma::inv(Rinv,R);
  arma::mat beta = W*Rinv*Q.t();
  arma::rowvec intercept = my - mx*beta;
  return List::create(_["coefficients"]=beta, 
                      _["intercept"]=NumericVector(intercept.begin(), intercept.end()),
                      _["x_weights"]=W, _["x_loadings"]=P, _["y_loadings"]=Q, _["scores"]=T,
                      _["x_means"]=NumericVector(mx.begin(), mx.end()), _["y_means"]=NumericVector(my.begin(), my.end()),
                      _["ncomp"]=used);
}