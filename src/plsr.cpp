#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo, bigmemory)]]

using namespace Rcpp;
using namespace arma;

#include <bigmemory/BigMatrix.h>
#include <bigmemory/MatrixAccessor.hpp>

// [[Rcpp::plugins(cpp17)]]

#include <algorithm>

namespace {

inline arma::mat compute_simpls(const arma::mat& Xc,
                                const arma::mat& Yc,
                                int ncomp,
                                arma::mat& weights,
                                arma::mat& loadings,
                                arma::mat& y_loadings,
                                arma::mat& scores)
{
    const arma::uword px = Xc.n_cols;
    const arma::uword q = Yc.n_cols;

    arma::mat S = Xc.t() * Yc;
    arma::mat R(px, ncomp, arma::fill::zeros);
    arma::mat P(px, ncomp, arma::fill::zeros);
    arma::mat Q(q, ncomp, arma::fill::zeros);
    arma::mat V(px, ncomp, arma::fill::zeros);

    int actual_comp = 0;
    arma::mat XtR;

    for (int a = 0; a < ncomp; ++a) {
        arma::vec r(px, arma::fill::zeros);
        arma::vec qvec(q, arma::fill::zeros);

        if (q == 1) {
            r = S.col(0);
            double norm_r = arma::norm(r, 2);
            if (norm_r <= 1e-12) {
                break;
            }
            r /= norm_r;
        } else {
            arma::vec svals;
            arma::mat U;
            arma::mat Vv;
            bool status = arma::svd_econ(U, svals, Vv, S);
            if (!status || svals.n_elem == 0 || svals[0] <= 1e-12) {
                break;
            }
            r = U.col(0);
        }

        arma::vec tvec = Xc * r;
        double norm_t = arma::norm(tvec, 2);
        if (norm_t <= 1e-12) {
            break;
        }
        arma::vec tnormed = tvec / norm_t;
        arma::vec pvec = Xc.t() * tnormed;
        qvec = Yc.t() * tnormed;

        S -= pvec * qvec.t();

        if (a < ncomp - 1) {
            arma::vec v = pvec;
            if (a > 0) {
                arma::mat Vsub = V.cols(0, a - 1);
                v -= Vsub * (Vsub.t() * pvec);
            }
            double vnorm = arma::norm(v, 2);
            if (vnorm > 1e-12) {
                v /= vnorm;
                V.col(a) = v;
                S -= v * (v.t() * S);
            }
        }

        R.col(a) = r;
        P.col(a) = pvec;
        Q.col(a) = qvec;
        ++actual_comp;
    }

    if (actual_comp == 0) {
        weights.reset();
        loadings.reset();
        y_loadings.reset();
        scores.reset();
        return arma::mat();
    }

    arma::mat R_used = R.cols(0, actual_comp - 1);
    arma::mat P_used = P.cols(0, actual_comp - 1);
    arma::mat Q_used = Q.cols(0, actual_comp - 1);

    arma::mat PtR = P_used.t() * R_used;
    arma::mat M = arma::inv(PtR);

    weights = R_used;
    loadings = P_used;
    y_loadings = Q_used;
    scores = Xc * R_used * M;

    return weights * M * Q_used.t();
}

inline void ensure_double_matrix(const BigMatrix& mat, const char* name)
{
    if (mat.matrix_type() != 8) {
        std::string msg = std::string(name) + " must be a double precision big.matrix";
        Rcpp::stop(msg);
    }
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List big_plsr_fit(SEXP X_ptr,
                        SEXP Y_ptr,
                        int ncomp,
                        bool center = true,
                        bool scale = false)
{
    if (ncomp <= 0) {
        Rcpp::stop("ncomp must be positive");
    }

    Rcpp::XPtr<BigMatrix> xMat(X_ptr);
    Rcpp::XPtr<BigMatrix> yMat(Y_ptr);

    ensure_double_matrix(*xMat, "X");
    ensure_double_matrix(*yMat, "Y");

    const arma::uword n = xMat->nrow();
    const arma::uword px = xMat->ncol();
    const arma::uword q = yMat->ncol();

    if (yMat->nrow() != n) {
        Rcpp::stop("X and Y must have the same number of rows");
    }
    if (n == 0) {
        Rcpp::stop("Input matrices must contain at least one row");
    }

    arma::mat X(static_cast<double*>(xMat->matrix()), n, px, false, true);
    arma::mat Y(static_cast<double*>(yMat->matrix()), n, q, false, true);

    arma::rowvec meanX(px, arma::fill::zeros);
    arma::rowvec meanY(q, arma::fill::zeros);
    arma::rowvec scaleX(px, arma::fill::ones);
    arma::rowvec scaleY(q, arma::fill::ones);

    arma::mat Xc = X;
    arma::mat Yc = Y;

    if (center) {
        meanX = arma::mean(Xc, 0);
        meanY = arma::mean(Yc, 0);
        Xc.each_row() -= meanX;
        Yc.each_row() -= meanY;
    }

    if (scale) {
        scaleX = arma::stddev(Xc, 0, 0);
        scaleY = arma::stddev(Yc, 0, 0);
        scaleX.replace(0.0, 1.0);
        scaleY.replace(0.0, 1.0);
        Xc.each_row() /= scaleX;
        Yc.each_row() /= scaleY;
    }

    arma::mat weights;
    arma::mat loadings;
    arma::mat y_loadings;
    arma::mat scores;

    arma::mat coef_internal = compute_simpls(Xc, Yc,
                                             std::min<int>(ncomp, static_cast<int>(px)),
                                             weights, loadings, y_loadings, scores);

    if (coef_internal.n_elem == 0) {
        return Rcpp::List::create(
            Rcpp::Named("coefficients") = R_NilValue,
            Rcpp::Named("intercept") = R_NilValue,
            Rcpp::Named("weights") = R_NilValue,
            Rcpp::Named("loadings") = R_NilValue,
            Rcpp::Named("y_loadings") = R_NilValue,
            Rcpp::Named("scores") = R_NilValue,
            Rcpp::Named("x_means") = meanX,
            Rcpp::Named("y_means") = meanY,
            Rcpp::Named("x_scales") = scaleX,
            Rcpp::Named("y_scales") = scaleY
        );
    }

    arma::mat coef = coef_internal;

    if (scale) {
        arma::vec scaleX_vec = scaleX.t();
        arma::vec scaleY_vec = scaleY.t();
        scaleX_vec.transform([](double val) { return 1.0 / val; });
        arma::mat scaleXInv = arma::diagmat(scaleX_vec);
        arma::mat scaleYDiag = arma::diagmat(scaleY_vec);
        coef = scaleXInv * coef * scaleYDiag;
    }

    arma::rowvec intercept = meanY;
    if (center) {
        intercept = meanY - meanX * coef;
    }

    return Rcpp::List::create(
        Rcpp::Named("coefficients") = coef,
        Rcpp::Named("intercept") = intercept,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("loadings") = loadings,
        Rcpp::Named("y_loadings") = y_loadings,
        Rcpp::Named("scores") = scores,
        Rcpp::Named("x_means") = meanX,
        Rcpp::Named("y_means") = meanY,
        Rcpp::Named("x_scales") = scaleX,
        Rcpp::Named("y_scales") = scaleY
    );
}

namespace {

arma::mat diagmat_from_vector(const arma::vec& v)
{
    arma::mat diag(v.n_elem, v.n_elem, arma::fill::zeros);
    for (arma::uword i = 0; i < v.n_elem; ++i) {
        diag(i, i) = v[i];
    }
    return diag;
}

} // anonymous namespace

// [[Rcpp::export]]
Rcpp::List big_plsr_stream_fit(SEXP X_ptr,
                               SEXP Y_ptr,
                               int ncomp,
                               bool center = true,
                               bool scale = false,
                               std::size_t block_size = 1024)
{
    if (ncomp <= 0) {
        Rcpp::stop("ncomp must be positive");
    }
    if (block_size == 0) {
        Rcpp::stop("block_size must be strictly positive");
    }

    Rcpp::XPtr<BigMatrix> xMat(X_ptr);
    Rcpp::XPtr<BigMatrix> yMat(Y_ptr);

    ensure_double_matrix(*xMat, "X");
    ensure_double_matrix(*yMat, "Y");

    const std::size_t n = xMat->nrow();
    const std::size_t px = xMat->ncol();
    const std::size_t q = yMat->ncol();

    if (yMat->nrow() != static_cast<int>(n)) {
        Rcpp::stop("X and Y must have the same number of rows");
    }
    if (n == 0) {
        Rcpp::stop("Input matrices must contain at least one row");
    }

    ncomp = std::min<int>(ncomp, static_cast<int>(px));

    MatrixAccessor<double> Xacc(*xMat);
    MatrixAccessor<double> Yacc(*yMat);

    std::vector<double> meanX(px, 0.0);
    std::vector<double> meanY(q, 0.0);

    if (center) {
        for (std::size_t j = 0; j < px; ++j) {
            const double* col = Xacc[j];
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += col[i];
            }
            meanX[j] = sum / static_cast<double>(n);
        }
        for (std::size_t k = 0; k < q; ++k) {
            const double* col = Yacc[k];
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += col[i];
            }
            meanY[k] = sum / static_cast<double>(n);
        }
    }

    std::vector<double> scaleX(px, 1.0);
    std::vector<double> scaleY(q, 1.0);

    if (scale) {
        for (std::size_t j = 0; j < px; ++j) {
            const double* col = Xacc[j];
            double sumsq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double val = col[i];
                if (center) {
                    val -= meanX[j];
                }
                sumsq += val * val;
            }
            double s = std::sqrt(sumsq / std::max<std::size_t>(1, n - 1));
            if (s > 0.0) {
                scaleX[j] = s;
            }
        }
        for (std::size_t k = 0; k < q; ++k) {
            const double* col = Yacc[k];
            double sumsq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                double val = col[i];
                if (center) {
                    val -= meanY[k];
                }
                sumsq += val * val;
            }
            double s = std::sqrt(sumsq / std::max<std::size_t>(1, n - 1));
            if (s > 0.0) {
                scaleY[k] = s;
            }
        }
    }

    arma::mat S(px, q, arma::fill::zeros);

    std::vector<double> x_row(px);
    std::vector<double> y_row(q);

    for (std::size_t start = 0; start < n; start += block_size) {
        std::size_t end = std::min<std::size_t>(n, start + block_size);
        for (std::size_t i = start; i < end; ++i) {
            for (std::size_t j = 0; j < px; ++j) {
                double val = Xacc[j][i];
                if (center) {
                    val -= meanX[j];
                }
                if (scale) {
                    val /= scaleX[j];
                }
                x_row[j] = val;
            }
            for (std::size_t k = 0; k < q; ++k) {
                double val = Yacc[k][i];
                if (center) {
                    val -= meanY[k];
                }
                if (scale) {
                    val /= scaleY[k];
                }
                y_row[k] = val;
            }
            for (std::size_t j = 0; j < px; ++j) {
                double xv = x_row[j];
                double* Srow = S.memptr() + j;
                for (std::size_t k = 0; k < q; ++k) {
                    Srow[k * px] += xv * y_row[k];
                }
            }
        }
    }

    arma::mat R(px, ncomp, arma::fill::zeros);
    arma::mat P(px, ncomp, arma::fill::zeros);
    arma::mat Qmat(q, ncomp, arma::fill::zeros);
    arma::mat V(px, ncomp, arma::fill::zeros);

    int actual_comp = 0;

    for (int a = 0; a < ncomp; ++a) {
        arma::vec r(px, arma::fill::zeros);

        if (q == 1) {
            r = S.col(0);
            double norm_r = arma::norm(r, 2);
            if (norm_r <= 1e-12) {
                break;
            }
            r /= norm_r;
        } else {
            arma::vec svals;
            arma::mat U;
            arma::mat Vv;
            bool status = arma::svd_econ(U, svals, Vv, S);
            if (!status || svals.n_elem == 0 || svals[0] <= 1e-12) {
                break;
            }
            r = U.col(0);
        }

        arma::vec pvec(px, arma::fill::zeros);
        arma::vec qvec(q, arma::fill::zeros);
        double tnorm_sq = 0.0;

        for (std::size_t start = 0; start < n; start += block_size) {
            std::size_t end = std::min<std::size_t>(n, start + block_size);
            for (std::size_t i = start; i < end; ++i) {
                for (std::size_t j = 0; j < px; ++j) {
                    double val = Xacc[j][i];
                    if (center) {
                        val -= meanX[j];
                    }
                    if (scale) {
                        val /= scaleX[j];
                    }
                    x_row[j] = val;
                }
                for (std::size_t k = 0; k < q; ++k) {
                    double val = Yacc[k][i];
                    if (center) {
                        val -= meanY[k];
                    }
                    if (scale) {
                        val /= scaleY[k];
                    }
                    y_row[k] = val;
                }
                double tval = 0.0;
                for (std::size_t j = 0; j < px; ++j) {
                    tval += x_row[j] * r[j];
                }
                tnorm_sq += tval * tval;
                for (std::size_t j = 0; j < px; ++j) {
                    pvec[j] += x_row[j] * tval;
                }
                for (std::size_t k = 0; k < q; ++k) {
                    qvec[k] += y_row[k] * tval;
                }
            }
        }

        if (tnorm_sq <= 1e-12) {
            break;
        }

        double tnorm = std::sqrt(tnorm_sq);
        pvec /= tnorm;
        qvec /= tnorm;

        S -= pvec * qvec.t();

        if (a < ncomp - 1) {
            arma::vec v = pvec;
            if (a > 0) {
                arma::mat Vsub = V.cols(0, a - 1);
                v -= Vsub * (Vsub.t() * pvec);
            }
            double vnorm = arma::norm(v, 2);
            if (vnorm > 1e-12) {
                v /= vnorm;
                V.col(a) = v;
                S -= v * (v.t() * S);
            }
        }

        R.col(a) = r;
        P.col(a) = pvec;
        Qmat.col(a) = qvec;
        ++actual_comp;
    }

    if (actual_comp == 0) {
        return Rcpp::List::create(
            Rcpp::Named("coefficients") = R_NilValue,
            Rcpp::Named("intercept") = R_NilValue,
            Rcpp::Named("weights") = R_NilValue,
            Rcpp::Named("loadings") = R_NilValue,
            Rcpp::Named("y_loadings") = R_NilValue,
            Rcpp::Named("x_means") = Rcpp::NumericVector(meanX.begin(), meanX.end()),
            Rcpp::Named("y_means") = Rcpp::NumericVector(meanY.begin(), meanY.end()),
            Rcpp::Named("x_scales") = Rcpp::NumericVector(scaleX.begin(), scaleX.end()),
            Rcpp::Named("y_scales") = Rcpp::NumericVector(scaleY.begin(), scaleY.end())
        );
    }

    arma::mat R_used = R.cols(0, actual_comp - 1);
    arma::mat P_used = P.cols(0, actual_comp - 1);
    arma::mat Q_used = Qmat.cols(0, actual_comp - 1);

    arma::mat PtR = P_used.t() * R_used;
    arma::mat M = arma::inv(PtR);
    arma::mat coef_internal = R_used * M * Q_used.t();

    arma::vec scaleX_vec(px);
    arma::vec scaleY_vec(q);
    for (std::size_t j = 0; j < px; ++j) {
        scaleX_vec[j] = scaleX[j];
    }
    for (std::size_t k = 0; k < q; ++k) {
        scaleY_vec[k] = scaleY[k];
    }

    arma::mat coef = coef_internal;
    if (scale) {
        arma::vec inv_scaleX = scaleX_vec;
        inv_scaleX.transform([](double val) { return 1.0 / val; });
        arma::mat scaleXInv = diagmat_from_vector(inv_scaleX);
        arma::mat scaleYDiag = diagmat_from_vector(scaleY_vec);
        coef = scaleXInv * coef * scaleYDiag;
    }

    arma::rowvec meanX_row(px);
    arma::rowvec meanY_row(q);
    for (std::size_t j = 0; j < px; ++j) {
        meanX_row[j] = meanX[j];
    }
    for (std::size_t k = 0; k < q; ++k) {
        meanY_row[k] = meanY[k];
    }

    arma::rowvec intercept = meanY_row;
    if (center) {
        intercept = meanY_row - meanX_row * coef;
    }

    return Rcpp::List::create(
        Rcpp::Named("coefficients") = coef,
        Rcpp::Named("intercept") = intercept,
        Rcpp::Named("weights") = R_used,
        Rcpp::Named("loadings") = P_used,
        Rcpp::Named("y_loadings") = Q_used,
        Rcpp::Named("x_means") = Rcpp::NumericVector(meanX.begin(), meanX.end()),
        Rcpp::Named("y_means") = Rcpp::NumericVector(meanY.begin(), meanY.end()),
        Rcpp::Named("x_scales") = Rcpp::NumericVector(scaleX.begin(), scaleX.end()),
        Rcpp::Named("y_scales") = Rcpp::NumericVector(scaleY.begin(), scaleY.end()),
        Rcpp::Named("ncomp") = actual_comp
    );
}