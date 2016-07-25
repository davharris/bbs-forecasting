library(rstan)
library(mvtnorm)

N = 50
x = c(scale(1:N))
y = rpois(N, exp(4 - x^2 / 5))
y = c(scale(y))
plot(y ~ x)


d = as.matrix(dist(x, method = "manhattan"))

data = list(x = x, y = y, N = N, d = d)



m = stan_model(
  "stan/shared-gp-core.stan"
)

model = sampling(m, data = data, control = list(adapt_delta = 0.95))

e = extract(model, permute = FALSE)


i = sample.int(4000, 1)
matplot(
  extract(model, "mu")[[1]][i] + t(rmvnorm(3, sigma = extract(model, "fn_variance")[[1]][i] * extract(model, "Sigma")[[1]][i,,])),
  type = "l",
  lty = 1,
  ylim =c(-4, 4)
)
abline(h = 0, col = "#00000050")
abline(h = extract(model, "mu")[[1]][i], col = "#00000020", lty = 2)
extract(model, "fn_variance")[[1]][i] + extract(model, "nugget")[[1]][i]


plot(density(extract(model, "fn_variance")[[1]], adj = .1), col = 2, main = "fn_variance")
curve(dgamma(x, 1.1, 0.1),from = 0, n = 1000, add = TRUE)


plot(density(extract(model, "lengthscale")[[1]], adj = .1), col = 2, main = "lengthscale")
curve(dgamma(x, 1.1, 0.1),from = 0, n = 1000, add = TRUE)

plot(density(extract(model, "nugget")[[1]], n = 1000), col = 2, main = "nugget")
curve(dgamma(x, 1.1, 0.1),from = 0, n = 1000, add = TRUE)

plot(density(extract(model, "mu")[[1]], n = 1000), col = 2, main = "mu")
curve(dcauchy(x, 0, 5), n = 1000, add = TRUE)


f = function(i)extract(model, "mu")[[1]][i] + rmvnorm(1, sigma = extract(model, "fn_variance")[[1]][i] * extract(model, "Sigma")[[1]][i,,])
