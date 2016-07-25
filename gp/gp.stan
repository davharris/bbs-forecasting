data {
  int<lower=1> N;
  vector[N] x;
  vector[N] y;
  matrix[N,N] d;
}
parameters {
  real<lower=0> fn_variance;
  real<lower=0> lengthscale;
  real<lower=0> nugget;
  real mu;
}
transformed parameters {
  vector[N] mu_vec;
  matrix[N, N] L;
  corr_matrix[N] Sigma;
  matrix[N,N] nugget_matrix;

  Sigma = exp(-d / lengthscale);

  for (i in 1:N){
    for (j in 1:N){
      nugget_matrix[i,j] = i==j ? nugget : 0.0;
    }
  }

  L = cholesky_decompose(fn_variance * Sigma + nugget_matrix);

  for(i in 1:N){
    mu_vec[i] = mu;
  }
}
model {
  // Prior with peak at 1. Very broad, >95% of the weight between .00005 and 13
  fn_variance ~ gamma(1.25, 0.25);
  lengthscale ~ gamma(1.25, 0.25);
  nugget ~ gamma(1.25, 0.25);

  mu ~ cauchy(mean(y), 5);
  y ~ multi_normal_cholesky(mu_vec, L);
}


