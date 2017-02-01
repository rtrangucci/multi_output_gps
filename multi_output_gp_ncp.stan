data {
  // Model is for D outcomes that have been
  // observed at the same x locations 
  int<lower=1> t; // Number of time periods 
  int<lower=1> D; // Number of outcomes/outputs
  matrix[t,D] y; // Outcome
  real x[t]; // Predictor
}
parameters {
  real<lower=0> len_scale; // Common length-scale
  vector<lower=0>[D] alphas; // Scale of outcomes
  real<lower=0> sigma; // Scale of noise
  // Outcome correlation matrix
  cholesky_factor_corr[D] L_Omega;
  // Standardized latent GP
  matrix[t, D] y_tilde;
}
transformed parameters {
  matrix[t, D] latent_gp; 
  {
    matrix[t, t] K;
    matrix[t, t] L_K;
    K = cov_exp_quad(x, 1.0, len_scale);
    for (n in 1:t)
      K[n,n] = K[n,n] + 1e-12;
    L_K = cholesky_decompose(K);

    // latent_gp is matrix-normal with among-column covariance K
    // among-row covariance multiply_lower_tri_self(alphas * L_Omega)
    
    latent_gp = L_K * y_tilde * diag_pre_multiply(alphas, L_Omega)';
  }
}
model {
  // priors
  sigma ~ normal(0, 1);
  len_scale ~ gamma(8, 2);
  alphas ~ normal(0, 1);
  L_Omega ~ lkj_corr_cholesky(3);
  to_vector(y_tilde) ~ normal(0, 1);

  // likelihood
  for (d in 1:D)
    y[,d] ~ normal(latent_gp[,d], sigma);
}
generated quantities {
  matrix[D, D] Omega;

  Omega = L_Omega * L_Omega';
}
