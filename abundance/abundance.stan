data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> N_species;

  matrix[N,P] x;
  int<lower=0> y[N_species, N];
}
parameters {
  real alpha_zero[N_species];
  real alpha_count[N_species];
  
  matrix[P, N_species] beta_zero;
  matrix[P, N_species] beta_count;
  
  vector[N] raw_noise;
  real sigma[N_species];
  real<lower=0> phi[N_species];
}
model {
  // Linear predictors
  matrix[N, N_species] x_beta_zero;
  matrix[N, N_species] x_beta_count;
  x_beta_zero = x * beta_zero;
  x_beta_count = x * beta_count;
  
  // Priors
  alpha_zero ~ logistic(0, 10);
  alpha_count ~ normal(0, 10);
  
  sigma ~ normal(0, 1);
  phi ~ exponential(1);
  
  for(i in 1:P){
    beta_zero[i] ~ normal(0, 1);
    beta_count[i] ~ normal(0, 1);
  }
  
  // Latent random variables
  raw_noise ~ normal(0, 1);
  
  // likelihood
  for (i in 1:N_species){
    for (j in 1:N) {
      real p_zero;
      real lambda;
      p_zero = inv_logit(alpha_zero[i] + x_beta_zero[j, i]);
      lambda = exp(alpha_count[i] + x_beta_count[j, i] + raw_noise[j] * sigma[i]);
      if (y[i, j] == 0) {
        // could be forced zero with probability p_zero or poisson-zero with
        // probability 1-p_zero
        target += log_sum_exp(bernoulli_lpmf(1 | p_zero),
                              bernoulli_lpmf(0 | p_zero)
                                + neg_binomial_2_lpmf(y[i, j] | lambda, phi[i]));
      } else {
        target += bernoulli_lpmf(0 | p_zero) + neg_binomial_2_lpmf(y[i, j] | lambda, phi[i]);
      }
    }
  }
}
