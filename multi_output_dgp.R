library(ggplot2)
library(rstan)

onion <- function(d, eta){
  shape1 <- rep(NA_real_, d - 1)
  shape2 <- rep(NA_real_, d - 1)
  alpha <- eta + (d - 2) / 2
  shape1[1] <- alpha
  shape2[1] <- alpha
  for(i in 2:(d - 1)){
    alpha <- alpha - 1 / 2
    shape2[i] <- alpha
    shape1[i] <- i / 2
  }
  r2 <- rbeta(d - 1, shape1, shape2)
  L <- matrix(0,d,d)
  L[1,1] <- 1
  L[2,1] <- 2 * r2[1] - 1
  L[2,2] <- sqrt(1 - L[2,1] ^ 2)
  for(m in 2:(d - 1)){
    l <- rnorm(m)
    scale <- sqrt(r2[m] / (t(l) %*% l)[1,])
    L[m + 1,1:m] <- l * scale
    L[m + 1, m + 1] <- sqrt(1 - r2[m])
  }
  return(L)
}


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

set.seed(320)

dim_x <- 100
D <- 3
alpha <- 1.0
len_scale <- 4

x <- seq(0,20,length.out = dim_x)
sds <- c(1,0.5,0.25)
L_inter_corr <- onion(D, .15)
inter_corr <- L_inter_corr %*% t(L_inter_corr)
cov_mat <- cov_exp_quad(x, alpha, len_scale, 1e-12)
L_mat <- t(chol(cov_mat))
y_tilde <- matrix(rnorm(D * dim_x),dim_x,D)
gp_draw <- L_mat %*% y_tilde %*% t(diag(sds) %*% L_inter_corr)
y <- gp_draw + matrix(0.5 * rnorm(D * dim_x), dim_x, D)

df <- data.frame(gp = as.vector(gp_draw),
                 y = as.vector(y),
                 series = as.vector(sapply(1:D,rep,dim_x)),
                 x = rep(x,D))

ggplot(df, aes(x = x, y = gp, colour = as.factor(series), group = series)) + geom_line() + geom_point(aes(x= x, y = y))

stan_dat <- list(y = y,
                 N = length(df$y),
                 t = dim_x,
                 D = D,
                 series = df$series,
                 x = x,
                 ind = as.vector(rep(1:dim_x,D)))

mod <- stan_model('multi_output_gp_ncp.stan')

fit <- sampling(mod, data = stan_dat, chains = 4, iter = 2000, control = list(adapt_delta = 0.95, max_treedepth = 15), cores = 4)

samps <- rstan::extract(fit)

print(fit, pars = c('alphas','sigma','len_scale','Omega'))
