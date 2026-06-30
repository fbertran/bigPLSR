# Simulated dataset

This dataset provides explantory variables simulations and censoring
status.

## Format

A data frame with 1000 observations on the following 11 variables.

- status:

  a binary vector

- X1:

  a numeric vector

- X2:

  a numeric vector

- X3:

  a numeric vector

- X4:

  a numeric vector

- X5:

  a numeric vector

- X6:

  a numeric vector

- X7:

  a numeric vector

- X8:

  a numeric vector

- X9:

  a numeric vector

- X10:

  a numeric vector

## References

TODO.  

## Examples

``` r

# \donttest{
data(sim_data)
X_sim_data_train <- sim_data[1:800,2:11]
C_sim_data_train <- sim_data$status[1:800]
X_sim_data_test <- sim_data[801:1000,2:11]
C_sim_data_test <- sim_data$status[801:1000]
rm(X_sim_data_train,C_sim_data_train,X_sim_data_test,C_sim_data_test)
# }
```
