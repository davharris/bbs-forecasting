library(rstan)
library(mvtnorm)

f = function(x) {2 * sin(pi *  x / 2)}

N1 = 25
N2 = 26
N = N1 + N2
x = c(scale(1:N))
y = rnorm(N, f(x), sd = 1)
y1 = y[1:N1]

y  = y  / sd(y1)
y1 = y1 / sd(y1)
plot(y ~ x)
abline(v = x[N1])


d = as.matrix(dist(x, method = "manhattan"))

data = list(N1 = N1, N2 = N2, x = x, y1 = y1, d = d, prior_mu_scale = 5)



m = stan_model(
  "gp/gp.stan"
)

model = sampling(m, data = data, control = list(adapt_delta = 0.95))



plot(density(extract(model, "fn_sd")[[1]], adj = .1), col = 2, main = "fn_sd")
curve(dgamma(x, 3, 1),from = 0, n = 1000, add = TRUE)


plot(density(extract(model, "lengthscale")[[1]], adj = 1, n = 1000), col = 2, main = "lengthscale", xlim = c(0, 100), xaxs = "i")
curve(dcauchy(x, 0, 2), n = 1000, add = TRUE)

plot(density(extract(model, "nugget_sd")[[1]], n = 1000), col = 2, main = "nugget")
curve(dgamma(x, 2, sqrt(2)),from = 0, n = 1000, add = TRUE)

plot(density(extract(model, "mu")[[1]], n = 1000), col = 2, main = "mu")
curve(dcauchy(x, 0, 5), n = 1000, add = TRUE)


y2s = extract(model, "y2")[[1]]
matplot(x[seq(N1 + 1, N)], t(y2s[1:1000, ]), type = "l", lty = 1, col = "#00000010", xlim = range(x))
curve(f(x), col = "darkred", lwd = 2, add = TRUE)
points(y ~ x, col = 2, pch = 16, cex = 1/2)
matlines(x[seq(N1 + 1, N)], t(apply(y2s, 2, quantile, c(.025, .975))), col = "purple", lty = 2, lwd = 2)

