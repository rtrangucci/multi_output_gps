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

neg_bi_comp <- stan_model('neg_bi_rng.stan')
neg_bi_funcs <- expose_stan_functions(neg_bi_comp)

## DGP

set.seed(320)

dim_x_1 <- 21
dim_x_2 <- 22
dim_x_3 <- 50
alpha_x_1_x_2 <- 0.25
alpha_x_3 <- 0.5
len_scale_x_1 <- 2
len_scale_x_2 <- 1
len_scale_x_3 <- 3
inv_sigma <- 3

x_1 <- seq(0,20,length.out = dim_x_1)
x_2 <- seq(0,20,length.out = dim_x_2)
x_3 <- seq(0,20,length.out = dim_x_3)
cov_mat_x_1 <- cov_exp_quad(x_1, 1.0, len_scale_x_1, 1e-12)
cov_mat_x_2 <- cov_exp_quad(x_2, alpha_x_1_x_2, len_scale_x_2, 1e-12)
cov_mat_x_3 <- cov_exp_quad(x_3, alpha_x_3, len_scale_x_3, 1e-12)
L_x_1 <- t(chol(cov_mat_x_1))
L_x_2 <- t(chol(cov_mat_x_2))
L_x_3 <- t(chol(cov_mat_x_3))
y_tilde <- array(rnorm(dim_x_1 * dim_x_2 * dim_x_3),dim = c(dim_x_1,dim_x_2, dim_x_3))
gp_draw <- array(NA_real_,dim = c(dim_x_1,dim_x_2, dim_x_3))
for (i in 1:dim_x_2) 
  gp_draw[,i,] <- y_tilde[,i,] %*% t(L_x_3)
for (i in 1:dim_x_3)
  gp_draw[,,i] <- L_x_1 %*% gp_draw[,,i] %*% t(L_x_2)
y <- apply(gp_draw,c(1,2,3),function(x) neg_bi_2_log_rng(x, inv_sigma))

## Generate data to pass to Stan model

stan_dat <- list(y = y,
                 dim_x_1 = dim_x_1,
                 dim_x_2 = dim_x_2,
                 dim_x_3 = dim_x_3,
                 x_1 = x_1,
                 x_2 = x_2,
                 x_3 = x_3)

mod <- stan_model('gp_3dim_ncp_NB.stan')

fit <- sampling(mod, data = stan_dat, chains = 4, iter = 1000, cores = 4, refresh = 100)

saveRDS(fit, file = 'gp_3dim_ncp_NB_21x22x50_fit.RDS')
samps <- rstan::extract(fit, pars = c('alpha', 'latent_gp', 'inv_sigma','len_scale_x_1','len_scale_x_2','len_scale_x_3'))

print(fit, pars = c('alpha', 'inv_sigma','len_scale_x_1','len_scale_x_2','len_scale_x_3'))
