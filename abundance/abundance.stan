data {
  int<lower=1> N;
  int<lower=1> P;

  matrix[N,P] x;
  int<lower=0> y[N];
}
parameters {
  real alpha_zero;
  real alpha_count;
  
  vector[P] beta_zero;
  vector[P] beta_count;
  
  vector[N] raw_noise;
  real<lower=0> sigma;
}
transformed parameters {
  vector[N] logit_p_zero;
  vector[N] lambda;
  logit_p_zero = alpha_zero + x * beta_zero;
  lambda = exp(alpha_count + x * beta_count + raw_noise * sigma);
}
model {
  // Priors
  alpha_zero ~ logistic(0, 10);
  beta_zero ~ logistic(0, 1);

  alpha_count ~ normal(0, 10);
  beta_count ~ normal(0, 1);
  sigma ~ gamma(2, .01);

  raw_noise ~ normal(0, 1);
  
  // likelihood
  for (i in 1:N) {
    real p_zero;
    p_zero = inv_logit(logit_p_zero[i]);
    
    if (y[i] == 0) {
      // could be forced zero with probability p_zero or poisson-zero with
      // probability 1-p_zero
      target += log_sum_exp(bernoulli_lpmf(1 | p_zero),
                            bernoulli_lpmf(0 | p_zero)
                              + poisson_lpmf(y[i] | lambda[i]));
    } else {
      target += bernoulli_lpmf(0 | p_zero) + poisson_lpmf(y[i] | lambda[i]);
    }
  }
}
generated quantities {
  vector[N] predicted_lambda;
  predicted_lambda = exp(alpha_count + x * beta_count + normal_rng(0, sigma));
}
