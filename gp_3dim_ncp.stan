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
  matrix[dim_x_1,dim_x_2] y[dim_x_3]; // Outcome
  real x_1[dim_x_1]; // locations - 1st dimension
  real x_2[dim_x_2]; // locations - 2nd dimension
  real x_3[dim_x_3]; // locations - 3rd dimension
}
parameters {
  real<lower=0> len_scale_x_1; // length-scale - 1st dimension
  real<lower=0> len_scale_x_2; // length-scale - 2nd dimension
  real<lower=0> len_scale_x_3; // length-scale - 3rd dimension
  real<lower=0> alpha; // Scale of outcomes
  real<lower=0> sigma; // Scale of noise
  // Standardized latent GP
  matrix[dim_x_1, dim_x_2] y_tilde[dim_x_3];
}
model {
  matrix[dim_x_1, dim_x_2] latent_gp[dim_x_3]; 
  {
    matrix[dim_x_1, dim_x_1] L_K_x_1 = gp_exp_quad_chol(x_1, 1.0, len_scale_x_1, 1e-12);
    matrix[dim_x_2, dim_x_2] t_L_K_x_2 = gp_exp_quad_chol(x_2, 1.0, len_scale_x_2, 1e-12)';
    matrix[dim_x_3, dim_x_3] L_K_x_3 = gp_exp_quad_chol(x_3, alpha, len_scale_x_3, 1e-12);

    // latent_gp is matrix-normal with among-column covariance K_x_1
    // among-row covariance K_x_2
    for (i in 1:dim_x_2)
      for(j in 1:dim_x_1) 
        latent_gp[,j,i] = to_array_1d(L_K_x_3 * to_vector(y_tilde[,j,i]));
    
    for (i in 1:dim_x_3)
      latent_gp[i] = L_K_x_1 * latent_gp[i] * t_L_K_x_2;
  }
  // priors
  sigma ~ normal(0, 1);
  len_scale_x_1 ~ gamma(8, 2);
  len_scale_x_2 ~ gamma(8, 2);
  len_scale_x_3 ~ gamma(8, 2);
  alpha ~ normal(0, 1);
  
  // likelihood
  for (i in 1:dim_x_3) {
    to_vector(y_tilde[i]) ~ normal(0, 1);
    to_vector(y[i]) ~ normal(to_vector(latent_gp[i]), sigma);
  }
}
