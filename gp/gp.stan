data {
  int<lower=1> N1;
  int<lower=1> N2;
  vector<lower=1>[N1 + N2] x;
  vector<lower=1>[N1] y1;
  matrix<lower=0>[N1 + N2, N1 + N2] d;

  real<lower=0> means_prior[2];
  real<lower=0> fn_scale_prior[2];
  real<lower=0> nugget_prior[2];
  real<lower=0> lengthscale_prior[2];

  real<lower=0> full_sd;
}
transformed data {
  int<lower=2> N;
  N = N1 + N2;
}
parameters {
  real<lower=0> lengthscale;
  vector[N2] y2;
  real mu;
  real<lower=0> nugget_sd;
  real<lower=0> fn_sd;
}
transformed parameters {
  vector[N] mu_vec;
  matrix[N,N] L;
  matrix<lower=0>[N,N] Sigma;
  matrix<lower=0>[N,N] nugget_matrix;

  Sigma = exp(-d / lengthscale);

  for (i in 1:N){
    for (j in 1:N){
      // //Saved for later: Matern 3/2
      // Sigma[i,j] = (1.0 +
      //               sqrt(3) * d[i,j] / lengthscale) *
      //                 exp(-sqrt(3) * d[i,j] / lengthscale);
      nugget_matrix[i,j] = i==j ? nugget_sd^2 : 0.0;
    }
  }

  L = cholesky_decompose(fn_sd^2 * Sigma + nugget_matrix);

  for(i in 1:N){
    mu_vec[i] = mu;
  }
}
model {
  // Bookkeeping //
  vector[N] y;
  for (n in 1:N1) y[n] = y1[n];
  for (n in 1:N2) y[N1 + n] = y2[n];

  // priors //
  mu ~ normal(means_prior[1], means_prior[2]); // Global mean
  mu ~ normal(mean(y1), means_prior[2]);       // Observed mean

  fn_sd ~ gamma(fn_scale_prior[1], fn_scale_prior[2]);
  lengthscale ~ gamma(lengthscale_prior[1], lengthscale_prior[2]);
  nugget_sd ~ normal(nugget_prior[1], nugget_prior[2]);

  // likelihood //
  y ~ multi_normal_cholesky(mu_vec, L);
}
