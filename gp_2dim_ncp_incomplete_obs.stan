functions {
  matrix gp_exp_quad_chol(real[] x, real alpha, real len, real jitter) {
    int dim_x = size(x);
    matrix[dim_x, dim_x] L_K_x;
    matrix[dim_x, dim_x] K_x = cov_exp_quad(x, alpha, len);
    for (n in 1:dim_x)
      K_x[n,n] = K_x[n,n] + jitter;
    L_K_x = cholesky_decompose(K_x);
    return L_K_x;
  }
}
data {
  // Model is for outcomes that have been
  // observed on a 2 dimensional grid
  int<lower=1> dim_x_1; // size of grid dimension 1
  int<lower=1> dim_x_2; // size of grid dimension 2
  int<lower=1> N; // Number of observations
  vector[N] y; // Outcome
  int row_i[N]; // row position of each observation
  int col_j[N]; // col position of each observation
  real x_1[dim_x_1]; // locations - 1st dimension
  real x_2[dim_x_2]; // locations - 2nd dimension
}
parameters {
  real<lower=0> len_scale_x_1; // length-scale - 1st dimension
  real<lower=0> len_scale_x_2; // length-scale - 2nd dimension
  real<lower=0> alpha; // Scale of outcomes
  real<lower=0> sigma; // Scale of noise
  // Standardized latent GP
  matrix[dim_x_1, dim_x_2] y_tilde;
}
model {
  matrix[dim_x_1, dim_x_2] latent_gp; 
  {
    matrix[dim_x_1, dim_x_1] L_K_x_1 = gp_exp_quad_chol(x_1, 1.0, len_scale_x_1, 1e-12);
    matrix[dim_x_2, dim_x_2] L_K_x_2 = gp_exp_quad_chol(x_2, alpha, len_scale_x_2, 1e-12);

    // latent_gp is matrix-normal with among-column covariance K_x_1
    // among-row covariance K_x_2
    
    latent_gp = L_K_x_1 * y_tilde * L_K_x_2';
  }
  // priors
  sigma ~ normal(0, 1);
  len_scale_x_1 ~ gamma(8, 2);
  len_scale_x_2 ~ gamma(8, 2);
  alpha ~ normal(0, 1);
  to_vector(y_tilde) ~ normal(0, 1);
  
  // likelihood
  y ~ normal(latent_gp[row_i, col_j], sigma);
}
