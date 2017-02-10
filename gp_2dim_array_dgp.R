library(ggplot2)
library(rstan)

## exponentiated quadratic covariance matrix function
cov_exp_quad <- function(x, alpha, len_scale, jitter) {
  dim_x = length(x)
  cov_ret <- matrix(NA_real_, dim_x, dim_x)
  for (i in 1:(dim_x - 1)) {
    cov_ret[i, i] = alpha ^ 2 + jitter
    for (j in (i + 1):dim_x) {
      r_x = (x[i] - x[j])^2
      cov_ret[i, j] = alpha^2 * exp(-1 / (2 * len_scale^2) * r_x)
      cov_ret[j,i] = cov_ret[i, j]
    }
  }
  cov_ret[dim_x, dim_x] = alpha ^ 2 + jitter
  return(cov_ret)
}

## DGP

set.seed(320)

dim_x_1 <- 100
dim_x_2 <- 100
alpha <- 0.5
len_scale_x_1 <- 4
len_scale_x_2 <- 2

x_1 <- seq(0,20,length.out = dim_x_1)
x_2 <- seq(0,20,length.out = dim_x_2)
cov_mat_x_1 <- cov_exp_quad(x_1, 1.0, len_scale_x_1, 1e-12)
cov_mat_x_2 <- cov_exp_quad(x_2, alpha, len_scale_x_2, 1e-12)
L_x_1 <- t(chol(cov_mat_x_1))
L_x_2 <- t(chol(cov_mat_x_2))
y_tilde <- matrix(rnorm(dim_x_1 * dim_x_2),dim_x_1,dim_x_2)
gp_draw <- L_x_1 %*% y_tilde %*% t(L_x_2)
y <- gp_draw + matrix(0.5 * rnorm(dim_x_1 * dim_x_2), dim_x_1, dim_x_2)

## Generate data to pass to Stan model

stan_dat <- list(y = y,
                 N = prod(dim(y)),
                 dim_x_1 = dim_x_1,
                 dim_x_2 = dim_x_2,
                 x_1 = x_1,
                 x_2 = x_2)

mod <- stan_model('gp_2dim_ncp.stan')

fit <- sampling(mod, data = stan_dat, chains = 4, iter = 2000, control = list(adapt_delta = 0.95, max_treedepth = 15), cores = 4)

samps <- rstan::extract(fit)

print(fit, pars = c('alpha','sigma','len_scale_x_1','len_scale_x_2'))
