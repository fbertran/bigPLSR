# Kernel Logistic PLS

## Kernel Logistic PLS (klogitpls)

We first extract latent scores with Kernel PLS (KPLS):

``` math
T = K_c U,
```

where $`K_c = H K(X,X) H`$ is the centered Gram matrix and the columns
of $`U`$ are the dual score directions (KPLS deflation).

We then fit a logistic link in the latent space using IRLS:

``` math
\eta = \beta_0 + T \beta, \qquad p = \sigma(\eta),
```
``` math
W = \mathrm{diag}(p (1-p)), \qquad z = \eta + \frac{y - p}{p(1-p)}.
```

At each iteration, solve the weighted least squares system for
$`[\beta_0, \beta]`$:
``` math
(\tilde{M}^\top \tilde{M}) \theta = \tilde{M}^\top \tilde{z}, \quad \tilde{M} = W^{1/2}[1, T], \ \tilde{z} = W^{1/2} z.
```

Optionally, we **alternate**: replace $`y`$ by $`p`$ and recompute KPLS
to refresh $`T`$ for a few steps.  
Prediction on new data uses the centered cross-kernel \$K_c(X\_\\, X)\$
and the stored KPLS basis $`U`$: \$\$ T\_\\ = K_c(X\_\\, X) \\ U, \qquad
\hat{p}\_\\ = \sigma\\\big(\beta_0 + T\_\\ \beta\big). \$\$
