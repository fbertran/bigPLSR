#' bigPLSR-package
#'
#' Provides Partial least squares Regression for big data. It allows for missing data in the explanatory variables. Repeated k-fold cross-validation of such models using various criteria. Bootstrap confidence intervals constructions are also available.
#'
#' @aliases bigPLSR-package bigPLSR NULL
#' 
#' @references Maumy, M., Bertrand, F. (2023). PLS models and their extension for big data. 
#'   Joint Statistical Meetings (JSM 2023), Toronto, ON, Canada. 
#'   
#'   Maumy, M., Bertrand, F. (2023). bigPLS: Fitting and cross-validating 
#'   PLS-based Cox models to censored big data. BioC2023 — The Bioconductor 
#'   Annual Conference, Dana-Farber Cancer Institute, Boston, MA, USA. 
#'   Poster. https://doi.org/10.7490/f1000research.1119546.1  
#' 
# #' @seealso [big_plsR()] and [big_plsR_gd()]
#' 
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(60), nrow = 20)
#' y <- X[, 1] - 0.5 * X[, 2] + rnorm(20, sd = 0.1)
#' fit <- pls_fit(X, y, ncomp = 2, scores = "r", algorithm = "simpls")
#' head(pls_predict_response(fit, X, ncomp = 2))
#' 
"_PACKAGE"

#' @importFrom graphics abline arrows axis layout lines legend par plot segments text 
#' @importFrom grDevices dev.new hcl.colors
#' @importFrom stats as.formula cov is.empty.model model.matrix model.response model.weights predict qchisq quantile rexp runif setNames var
#' @importFrom utils modifyList read.csv
#' @import bigmemory
#' @useDynLib bigPLSR, .registration = TRUE
#' @importFrom Rcpp evalCpp
# #' @import bigalgebra
NULL
