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
#' set.seed(314)
#' library(bigPLSR)
#' data(sim_data)
#' head(sim_data)
#' 
"_PACKAGE"

#' @importFrom graphics abline arrows axis layout legend plot segments text 
#' @importFrom grDevices dev.new
#' @importFrom stats as.formula is.empty.model model.matrix model.response model.weights quantile rexp runif var
#' @importFrom utils read.csv
#' @import bigmemory
#' @useDynLib bigPLSR, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @import bigalgebra
NULL
