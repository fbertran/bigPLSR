// [[Rcpp::depends(RcppArmadillo, bigmemory)]]
// [[Rcpp::plugins(cpp17)]]
#include <RcppArmadillo.h>
#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>
#include <cstring>      // memcpy
#include <algorithm>    // std::min
using namespace Rcpp;

// Optional CBLAS fast path: include only if the header is available.
// This avoids including the Accelerate umbrella (which pulls vDSP and conflicts with R's COMPLEX).
#if defined(BIGPLSR_USE_CBLAS)
#if defined(__APPLE__)
#if __has_include(<vecLib/cblas.h>)
extern "C" { 
#include <vecLib/cblas.h> 
}
#else
// Header not present in SDK; fall back to the portable Armadillo path.
#undef BIGPLSR_USE_CBLAS
#endif
#else
#if __has_include(<cblas.h>)
extern "C" { 
#include <cblas.h> 
}
#else
#undef BIGPLSR_USE_CBLAS
#endif
#endif
#endif

// choose a cache-friendly default chunk size when chunk_size == 0
static inline std::size_t default_chunk_size(std::size_t n) {
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
  const std::size_t d = 16384;   // Apple Silicon: bigger chunk helps GEMM
#else
  const std::size_t d = 4096;
#endif
  return (n < d) ? n : d;
}

// ---- 1) Chunked cross-products from big.matrix (centered) ----
// [[Rcpp::export]]
SEXP cpp_bigmem_cross(SEXP X_ptrSEXP, SEXP Y_ptrSEXP, std::size_t chunk_size) {
  XPtr<BigMatrix> X_ptr(X_ptrSEXP);
  XPtr<BigMatrix> Y_ptr(Y_ptrSEXP);
  
  MatrixAccessor<double> X_acc(*X_ptr);
  MatrixAccessor<double> Y_acc(*Y_ptr);
  
  const std::size_t n = static_cast<std::size_t>(X_ptr->nrow());
  const std::size_t p = static_cast<std::size_t>(X_ptr->ncol());
  const std::size_t m = static_cast<std::size_t>(Y_ptr->ncol());
  if ((std::size_t)Y_ptr->nrow() != n) stop("X and Y must have matching rows");
  if (n == 0 || p == 0 || m == 0) stop("empty X or Y");
  
  arma::vec sumX(p, arma::fill::zeros);
  arma::rowvec sumY(m, arma::fill::zeros);
  arma::mat sumXX(p, p, arma::fill::zeros);
  arma::mat sumXY(p, m, arma::fill::zeros);
  
  const std::size_t chunk = (chunk_size > 0) ? chunk_size : default_chunk_size(n);
  const std::size_t max_chunk = std::min<std::size_t>(chunk, n);
  // pre-allocate and reuse row buffers
  arma::mat B_buf(max_chunk, p);
  arma::mat Yb_buf(max_chunk, m);
  
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    const std::size_t r   = end - start;
    
    // views on the reusable buffers limited to first r rows (for sums / arma fallback)
    auto Brow = B_buf.rows(0, r - 1);
    auto Yrow = Yb_buf.rows(0, r - 1);
    // fill columns via memcpy into the first r rows
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      std::memcpy(B_buf.colptr(j), col + start, r * sizeof(double));
    }
    for (std::size_t k = 0; k < m; ++k) {
      const double* col = Y_acc[k];
      std::memcpy(Yb_buf.colptr(k), col + start, r * sizeof(double));
    }
    
    // accumulate sums
    sumX += arma::sum(Brow, 0).t();
    sumY += arma::sum(Yrow, 0);
#if defined(BIGPLSR_USE_CBLAS) && defined(BIGPLSR_HAVE_CBLAS)
    // In-place CBLAS accumulations using parent buffers + leading dimensions:
    // sumXX (p x p) += (B'B) with k = r, lda/ldb = max_chunk
{
  const int p_i   = static_cast<int>(p);
  const int r_i   = static_cast<int>(r);
  const int m_i   = static_cast<int>(m);
  const int ldc_x = static_cast<int>(p);           // sumXX leading dim
  const int ldc_y = static_cast<int>(p);           // sumXY leading dim
  const int ldB   = static_cast<int>(max_chunk);   // physical rows in B_buf / Yb_buf
  
  // C := 1.0 * (B_buf^T [p x r]) * (B_buf [r x p]) + 1.0 * C
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
              /*M=*/p_i, /*N=*/p_i, /*K=*/r_i,
              /*alpha=*/1.0,
              /*A=*/B_buf.memptr(), /*lda=*/ldB,
              /*B=*/B_buf.memptr(), /*ldb=*/ldB,
              /*beta=*/1.0,
              /*C=*/sumXX.memptr(), /*ldc=*/ldc_x);
              
              // C := 1.0 * (B_buf^T [p x r]) * (Yb_buf [r x m]) + 1.0 * C
              cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                          /*M=*/p_i, /*N=*/m_i, /*K=*/r_i,
                          /*alpha=*/1.0,
                          /*A=*/B_buf.memptr(), /*lda=*/ldB,
                          /*B=*/Yb_buf.memptr(),/*ldb=*/ldB,
                          /*beta=*/1.0,
                          /*C=*/sumXY.memptr(), /*ldc=*/ldc_y);
}
#else
    // Portable fallback (still BLAS-backed by Armadillo)
    arma::mat Bt = Brow.t();              // p x r
    sumXX += Bt * Brow;                   // p x p  (B'B)
    sumXY += Bt * Yrow;                   // p x m  (B'Y)
#endif
  }
  
  const double nd = static_cast<double>(n);
  arma::vec x_means = sumX / nd;
  arma::rowvec y_means = sumY / nd;
  
  arma::mat XtX = sumXX - nd * (x_means * x_means.t());
  arma::mat XtY = sumXY - nd * (x_means * y_means);
  
  return List::create(
    _["XtX"] = XtX,
    _["XtY"] = XtY,
    _["x_means"] = x_means,
    _["y_means"] = y_means
  );
}

// ---- 2) SIMPLS on cross-products (multi-response) ----
//   XtX: p x p, symmetric (centered cross-products)
//   XtY: p x m, centered cross-products
// [[Rcpp::export]]
SEXP cpp_simpls_from_cross(const arma::mat& XtX,
                           const arma::mat& XtY,
                           const arma::vec& x_mean,
                           const arma::rowvec& y_mean,
                           int ncomp,
                           double tol) {
  const std::size_t p = XtX.n_rows;
  const std::size_t m = XtY.n_cols;
  if (XtX.n_cols != p || XtY.n_rows != p) stop("dimension mismatch XtX/XtY");
  
  arma::mat XtX_curr = 0.5 * (XtX + XtX.t()); // symmetrize
  arma::mat XtY_curr = XtY;
  
  arma::mat W(p, ncomp, arma::fill::zeros);
  arma::mat P(p, ncomp, arma::fill::zeros);
  arma::mat Q(m, ncomp, arma::fill::zeros);
  
  int used = 0;
  for (int h = 0; h < ncomp; ++h) {
    // generalized eigen via Cholesky whitening
    // (XtY XtY^T) w = λ (XtX) w
    arma::mat A = XtY_curr * XtY_curr.t();  // p x p (theoretically symmetric)
    A = 0.5 * (A + A.t());                  // enforce symmetry numerically
    arma::mat B = XtX_curr;                 // p x p, SPD-ish
    const double ridge = std::max(tol, 1e-12) * (arma::trace(B) / std::max<double>(1.0, (double)p));
    B.diag() += ridge;
    
    arma::mat C;
    if (!arma::chol(C, B)) {
      // try a slightly larger ridge if needed
      B.diag() += std::max(1e-8, ridge * 10.0);
      if (!arma::chol(C, B)) break; // give up on this component
    }
    // C is upper s.t. B = C.t() * C; build K = C^{-T} A C^{-1}
    arma::mat Ci = arma::solve(arma::trimatu(C), arma::eye<arma::mat>(p, p)); // C^{-1}
    arma::mat K  = Ci.t() * A * Ci;                                           // should be symmetric
    K = 0.5 * (K + K.t());                                                    // enforce symmetry numerically
    
    arma::vec evals;
    arma::mat V;
    arma::eig_sym(evals, V, K);
    if (evals.n_elem == 0u) break;
    
    arma::vec v = V.col(V.n_cols - 1);    // dominant eigenvector of K
    arma::vec w = Ci * v;                 // generalized eigenvector
    double denom = arma::as_scalar(w.t() * XtX_curr * w);
    if (!std::isfinite(denom) || denom <= tol) break;
    w /= std::sqrt(denom);                // enforce w^T XtX w ≈ 1
    
    arma::vec XtXw = XtX_curr * w;              // p x 1
    arma::rowvec C_row = (w.t() * XtY_curr);    // 1 x m
    arma::vec p_vec = XtXw;                     // since denom ~ 1
    
    // store
    W.col(h) = w;
    P.col(h) = p_vec;
    Q.col(h) = C_row.t();
    
    // deflate
    arma::mat XtX_new = XtX_curr - XtXw * p_vec.t() - p_vec * XtXw.t() + (p_vec * p_vec.t());
    arma::rowvec WtXtY = C_row; // same as w.t() * XtY_curr
    arma::mat XtY_new = XtY_curr - XtXw * C_row - p_vec * WtXtY + (p_vec * C_row);
    
    XtX_curr = 0.5 * (XtX_new + XtX_new.t());
    XtY_curr = XtY_new;
    ++used;
  }
  
  if (used == 0) {
    return List::create(
      _["coefficients"] = arma::mat(p, m, arma::fill::zeros),
      _["intercept"]    = y_mean,
      _["x_weights"]    = arma::mat(p, 0),
      _["x_loadings"]   = arma::mat(p, 0),
      _["y_loadings"]   = arma::mat(m, 0),
      _["x_means"]      = x_mean,
      _["y_means"]      = y_mean,
      _["ncomp"]        = 0
    );
  }
  
  arma::mat W_sub = W.cols(0, used - 1);
  arma::mat P_sub = P.cols(0, used - 1);
  arma::mat Q_sub = Q.cols(0, used - 1);   // m x used
  
  arma::mat R = P_sub.t() * W_sub;         // used x used
  arma::mat coef = W_sub * arma::solve(R, Q_sub.t());   // p x m
  arma::rowvec intercept = y_mean - (x_mean.t() * coef); // 1 x m
  
  return List::create(
    _["coefficients"] = coef,
    _["intercept"]    = intercept,
    _["x_weights"]    = W_sub,
    _["x_loadings"]   = P_sub,
    _["y_loadings"]   = Q_sub,
    _["x_means"]      = x_mean,
    _["y_means"]      = y_mean,
    _["ncomp"]        = used
  );
}

// ---- 3) Stream scores: T = (X - 1*x_mean^T) * W (chunked) ----
// scores_sinkSEXP: NULL or big.matrix S4.
// return_big: if TRUE and sink is NULL -> allocate file-backed big.matrix via R-side; here we only write.
[[maybe_unused]] static inline S4 allocate_big_matrix(std::size_t nrow, std::size_t ncol, const char* name); 
// forward decl if you already have it elsewhere

// [[Rcpp::export]]
SEXP cpp_stream_scores_given_W(SEXP X_ptrSEXP,
                               const arma::mat& W,
                               const arma::vec& x_mean,
                               std::size_t chunk_size,
                               SEXP scores_sinkSEXP,
                               bool return_big) {
  XPtr<BigMatrix> X_ptr(X_ptrSEXP);
  MatrixAccessor<double> X_acc(*X_ptr);
  const std::size_t n = static_cast<std::size_t>(X_ptr->nrow());
  const std::size_t p = static_cast<std::size_t>(X_ptr->ncol());
  const std::size_t r = (std::size_t)W.n_cols;
  if ((std::size_t)W.n_rows != p) stop("W dimension mismatch with X");
  
  const std::size_t chunk = (chunk_size > 0) ? chunk_size : default_chunk_size(n);
  const std::size_t max_chunk = std::min<std::size_t>(chunk, n);
  arma::mat B_buf(max_chunk, p);  // reused
  
  // decide sink
  std::function<void(std::size_t, const arma::mat&)> write_scores;
  RObject scores_out;
  std::unique_ptr<arma::mat> T; bool to_inmemory = false;
  
  if (!Rf_isNull(scores_sinkSEXP)) {
    S4 sink_s4(scores_sinkSEXP);
    XPtr<BigMatrix> sink_ptr(sink_s4.slot("address"));
    if ((std::size_t)sink_ptr->nrow() != n || (std::size_t)sink_ptr->ncol() != r)
      stop("scores sink dimension mismatch");
    auto acc = std::make_shared<MatrixAccessor<double>>(*sink_ptr);
    write_scores = [acc, r](std::size_t start, const arma::mat& S) {
      const std::size_t m = S.n_rows;
      for (std::size_t h = 0; h < r; ++h) {
        std::memcpy((*acc)[(int)h] + start, S.colptr(h), m * sizeof(double));
      }
    };
    scores_out = sink_s4;
  } else if (return_big) {
    // leave allocation to R-side for file-backed; here we fall back to in-memory matrix
    T.reset(new arma::mat(n, r, arma::fill::zeros));
    write_scores = [Tptr=T.get()](std::size_t start, const arma::mat& S) {
      Tptr->rows(start, start + S.n_rows - 1) = S;
    };
    to_inmemory = true;
  } else {
    T.reset(new arma::mat(n, r, arma::fill::zeros));
    write_scores = [Tptr=T.get()](std::size_t start, const arma::mat& S) {
      Tptr->rows(start, start + S.n_rows - 1) = S;
    };
    to_inmemory = true;
  }
  
  arma::rowvec mu = x_mean.t();
  for (std::size_t start = 0; start < n; start += chunk) {
    const std::size_t end = std::min<std::size_t>(n, start + chunk);
    const std::size_t rc  = end - start;
    auto Brow = B_buf.rows(0, rc - 1);
    for (std::size_t j = 0; j < p; ++j) {
      const double* col = X_acc[j];
      std::memcpy(B_buf.colptr(j), col + start, rc * sizeof(double));
    }
    Brow.each_row() -= mu;
    arma::mat S = Brow * W; // rc x r
    write_scores(start, S);
  }
  
  if (to_inmemory) return wrap(*T);
  return scores_out; // an S4 big.matrix
}
