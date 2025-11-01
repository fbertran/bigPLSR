#' Simulated dataset
#' 
#' This dataset provides explantory variables simulations and censoring status.
#' 
#' 
#' @name sim_data
#' @docType data
#' @format A data frame with 1000 observations on the following 11 variables.
#' \describe{ 
#' \item{status}{a binary vector} 
#' \item{X1}{a numeric vector} 
#' \item{X2}{a numeric vector} 
#' \item{X3}{a numeric vector} 
#' \item{X4}{a numeric vector} 
#' \item{X5}{a numeric vector} 
#' \item{X6}{a numeric vector} 
#' \item{X7}{a numeric vector} 
#' \item{X8}{a numeric vector} 
#' \item{X9}{a numeric vector} 
#' \item{X10}{a numeric vector} 
#' }
#' @references TODO.\cr
#' 
#' @keywords datasets
#' @examples
#' 
#' \donttest{
#' data(sim_data)
#' X_sim_data_train <- sim_data[1:800,2:11]
#' C_sim_data_train <- sim_data$status[1:800]
#' X_sim_data_test <- sim_data[801:1000,2:11]
#' C_sim_data_test <- sim_data$status[801:1000]
#' rm(X_sim_data_train,C_sim_data_train,X_sim_data_test,C_sim_data_test)
#' }
#' 
NULL

