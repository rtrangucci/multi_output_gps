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
  // observed on a 3 dimensional grid
  int<lower=1> dim_x_1; // size of grid dimension 1
  int<lower=1> dim_x_2; // size of grid dimension 2
  int<lower=1> dim_x_3; // size of grid dimension 3
  int y[dim_x_1, dim_x_2, dim_x_3]; // Outcome
  real x_1[dim_x_1]; // locations - 1st dimension
  real x_2[dim_x_2]; // locations - 2nd dimension
  real x_3[dim_x_3]; // locations - 3rd dimension
}
parameters {
  real<lower=0> len_scale_x_1; // length-scale - 1st dimension
  real<lower=0> len_scale_x_2; // length-scale - 2nd dimension
  real<lower=0> len_scale_x_3; // length-scale - 3rd dimension
  real<lower=0> alpha_x_3; // Scale of outcomes
  real<lower=0> alpha_x_1_x_2; // Scale of outcomes
  real<lower=0> inv_phi;
  // Standardized latent GP
  real y_tilde[dim_x_1, dim_x_2, dim_x_3];
}
model {
  real latent_gp[dim_x_1, dim_x_2, dim_x_3];
  real phi = inv(inv_phi);
  {
    matrix[dim_x_1, dim_x_1] L_K_x_1 = gp_exp_quad_chol(x_1, 1.0, len_scale_x_1, 1e-12);
    matrix[dim_x_2, dim_x_2] t_L_K_x_2 = gp_exp_quad_chol(x_2, alpha_x_1_x_2, len_scale_x_2, 1e-12)';
    matrix[dim_x_3, dim_x_3] t_L_K_x_3 = gp_exp_quad_chol(x_3, alpha_x_3, len_scale_x_3, 1e-12)';

    // latent_gp is matrix-normal with among-column covariance K_x_1
    // among-row covariance K_x_2
    for (i in 1:dim_x_2)
      latent_gp[,i,] = to_array_2d(to_matrix(y_tilde[,i,]) * t_L_K_x_3);
    
    for (i in 1:dim_x_3)
      latent_gp[,,i] = to_array_2d(L_K_x_1 * to_matrix(latent_gp[,,i]) * t_L_K_x_2);
  }
  // priors
  len_scale_x_1 ~ gamma(8, 2);
  len_scale_x_2 ~ gamma(8, 2);
  len_scale_x_3 ~ gamma(8, 2);
  alpha_x_3 ~ normal(0, 1);
  alpha_x_1_x_2~ normal(0, 1);
  inv_phi ~ normal(0, 1);
  
  // likelihood
  to_array_1d(y_tilde) ~ normal(0, 1);
  to_array_1d(y) ~ neg_binomial_2_log(to_array_1d(latent_gp), phi);
}
