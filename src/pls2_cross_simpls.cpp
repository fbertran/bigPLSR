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

static inline void orthonormal_append(arma::mat& V, const arma::vec& v_new, double tol) {
  arma::vec v = v_new;
  if (V.n_cols > 0) {
    v -= V * (V.t() * v);
  }
  double nv = arma::norm(v, 2);
  if (nv > tol) {
    arma::vec vunit = v / nv;
    V.insert_cols(V.n_cols, vunit);
  }
}

// [[Rcpp::export]]
SEXP cpp_simpls_from_cross(const arma::mat& XtX,
                           const arma::mat& XtY,
                           const arma::vec& x_mean,
                           const arma::rowvec& y_mean,
                           int ncomp,
                           double tol) {
  if (ncomp <= 0) Rcpp::stop("ncomp must be positive");
  const arma::uword p = XtX.n_rows;
  const arma::uword m = XtY.n_cols;
  if (XtX.n_cols != p) Rcpp::stop("XtX must be square p x p");
  if (XtY.n_rows != p) Rcpp::stop("XtY must be p x m");
  
  // running containers
  arma::mat W(p, ncomp, arma::fill::zeros);
  arma::mat P(p, ncomp, arma::fill::zeros);
  arma::mat Q(m, ncomp, arma::fill::zeros);
  arma::mat V(p, 0, arma::fill::none); // orthonormal (Euclidean) basis
  
  arma::mat S = XtY;                   // will be deflated via V
  int a_used = 0;
  
  for (int a = 0; a < ncomp; ++a) {
    // Deflate S in subspace spanned by V (Euclidean-orthonormal columns)
    if (V.n_cols > 0) {
      S -= V * (V.t() * S);
    }
    // Leading eigenvector of S' S (m x m)
    arma::mat StS = S.t() * S;  // symmetric
    arma::vec evals;
    arma::mat evecs;
    if (!arma::eig_sym(evals, evecs, StS)) {
      break;
    }
    if (evals.n_elem == 0 || evals.max() <= tol) {
      break;
    }
    arma::uword idx = evals.index_max();
    arma::vec q = evecs.col(idx);         // m x 1
    
    // w = S q ; normalize in X-metric: sqrt(w' XtX w) = 1
    arma::vec w = S * q;                  // p x 1
    double xnorm = std::sqrt( arma::as_scalar(w.t() * XtX * w) );
    if (!arma::is_finite(xnorm) || xnorm <= tol) {
      break;
    }
    w /= xnorm;
    
    arma::vec pvec = XtX * w;             // p x 1
    arma::vec v    = pvec;                 // for V expansion
    // Euclidean orthonormalization of V with new column v
    {
      arma::vec vtmp = v;
      if (V.n_cols > 0) vtmp -= V * (V.t() * vtmp);
      double nv = arma::norm(vtmp, 2);
      if (nv <= tol) break;
      vtmp /= nv;
      V.insert_cols(V.n_cols, vtmp);
    }
    
    arma::vec qload = S.t() * w;          // m x 1
    
    W.col(a) = w;
    P.col(a) = pvec;
    Q.col(a) = qload;
    ++a_used;
  }
  
  if (a_used == 0) {
    return Rcpp::List::create(
      Rcpp::Named("coefficients") = R_NilValue,
      Rcpp::Named("intercept")    = y_mean,
      Rcpp::Named("x_weights")    = R_NilValue,
      Rcpp::Named("x_loadings")   = R_NilValue,
      Rcpp::Named("y_loadings")   = R_NilValue,
      Rcpp::Named("x_means")      = x_mean,
      Rcpp::Named("y_means")      = y_mean,
      Rcpp::Named("ncomp")        = 0
    );
  }
  
  arma::mat WU = W.cols(0, a_used-1);
  arma::mat PU = P.cols(0, a_used-1);
  arma::mat QU = Q.cols(0, a_used-1);
  
  // R = P' W  (small a_used x a_used); solve for R^{-1}
  arma::mat R = PU.t() * WU;
  arma::mat Rinv;
  if (!arma::inv(Rinv, R)) {
    Rcpp::stop("Failed to invert P'W in SIMPLS.");
  }
  // beta = W R^{-1} Q'
  arma::mat coef = WU * Rinv * QU.t();      // p x m
  
 // Ensure x_mean is a row for (1×p) · (p×m) → (1×m)
 arma::rowvec xmean_row;
 {
   // handle either vec or rowvec inputs safely
   if (x_mean.n_rows > 1 && x_mean.n_cols == 1) {
     xmean_row = x_mean.t();              // vec → rowvec
   } else if (x_mean.n_rows == 1) {
     xmean_row = arma::rowvec(x_mean);    // already row
   } else {
     xmean_row = x_mean.t();              // fallback
   }
 }
  arma::rowvec intercept = y_mean - xmean_row * coef;
    
  // intercept = y_mean - x_mean^T * coef
  // arma::rowvec intercept = y_mean - (x_mean.t().t() * coef); // keep rowvec shape
    
  
  return Rcpp::List::create(
    Rcpp::Named("coefficients") = coef,
    Rcpp::Named("intercept")    = intercept,
    Rcpp::Named("x_weights")    = WU,
    Rcpp::Named("x_loadings")   = PU,
    Rcpp::Named("y_loadings")   = QU,
    Rcpp::Named("x_means")      = x_mean,
    Rcpp::Named("y_means")      = y_mean,
    Rcpp::Named("ncomp")        = a_used
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
