data {
  int<lower=1> N1;
  int<lower=1> N2;
  vector[N1 + N2] x;
  vector[N1] y1;
  matrix[N1 + N2, N1 + N2] d;
  real prior_mu_scale;
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
  matrix[N,N] Sigma;
  matrix[N,N] nugget_matrix;

  Sigma = exp(-d / lengthscale);

  for (i in 1:N){
    for (j in 1:N){
      Sigma[i,j] = (1.0 +
                    sqrt(3) * d[i,j] / lengthscale) *
                      exp(-sqrt(3) * d[i,j] / lengthscale);
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

  // Lengthscale is probably on the order of 1 sd(x)
  lengthscale ~ student_t(10, 0, 5);

  // gamma(4, 1) prior has a peak around 3 sd (i.e. function's variance is about
  // 3^2 times as big as the variance of the observed portion)
  fn_sd ~ gamma(3.0, 1.0);

  // Nugget shouldn't be 0 or enormous. Prior peaked around sqrt(1/2)
  // corresponds to a nugget that accounts for half of the variance
  nugget_sd ~ gamma(2.0, sqrt(2.0));

  mu ~ cauchy(mean(y), 2.0);

  // likelihood //

  y ~ multi_normal_cholesky(mu_vec, L);
}
