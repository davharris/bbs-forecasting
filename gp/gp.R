library(rstan)
library(mvtnorm)

f = function(x) {2 * sin(pi *  x / 2) + sin(pi * x^2) + 1.2 * x + x^2}

N1 = 40
N2 = 41
N = N1 + N2
x = c(scale(1:N))
x[N] = 1000
y = rnorm(N, f(x), sd = 1)

plot(y[-N] ~ x[-N])
abline(v = x[N1])

y1 = y[1:N1]

shift = mean(y[1:N1])
std = sd(y[1:N1])

y  = (y - shift)  / std
y1 = (y1 - shift) / std


d = as.matrix(dist(x, method = "manhattan"))

data = list(N1 = N1, N2 = N2, x = x, y1 = y1, d = d, prior_mu_scale = 5)



m = stan_model(
  "gp/gp.stan"
)

model = sampling(m, data = data, control = list(adapt_delta = 0.95),
                 chains = 2)


plot(density(extract(model, "fn_sd")[[1]], adj = .1), col = 2, main = "fn_sd")
curve(dgamma(x, 3, 1),from = 0, n = 1000, add = TRUE)

plot(density(extract(model, "lengthscale")[[1]], adj = 0.1, n = 1000, from = 0), col = 2, main = "lengthscale", xaxs = "i")
curve(2 * dt(x/5, 10) / 5, n = 1000, from = 0, add = TRUE)

plot(density(extract(model, "nugget_sd")[[1]], n = 1000), col = 2, main = "nugget")
curve(dgamma(x, 2, sqrt(2)),from = 0, n = 1000, add = TRUE)

plot(density(extract(model, "mu")[[1]], n = 1000), col = 2, main = "mu")
curve(dcauchy(x, 0, 2), n = 1000, add = TRUE)


y2s = extract(model, "y2")[[1]]
matplot(x[seq(N1 + 1, N)], t(y2s[sample.int(nrow(y2s), 250), ]), type = "l", lty = 1, col = "#00000010", xlim = range(x[-N]), xaxs = "i", lwd = 2,
        ylim = quantile(y2s, c(.005, .995)))
curve(((f(x) - shift) / std) / std, col = "darkred", lwd = 2, add = TRUE, from = x[1], to = x[N-1])
lines(x[seq(N1 + 1, N)], colMeans(y2s), col = "blue", lwd = 2)
matlines(x[seq(N1 + 1, N)], t(apply(y2s, 2, quantile, c(.025, .975))), col = "purple", lty = 2, lwd = 2)
matlines(x[seq(N1 + 1, N)], t(apply(y2s, 2, quantile, c(pnorm(c(-1,1))))), col = "purple", lty = 2, lwd = 2)
abline(h = quantile(y2s[, N2], c(.025, .975)), lty = 2)



S = extract(model, "Sigma")[[1]]
mus = extract(model, "mu")[[1]]
scales = extract(model, "fn_sd")[[1]]

i = sample.int(nrow(S), 1)
joint = mus[i] + rmvnorm(5, sigma = S[i,,] * scales[i]^2)
matplot(t(joint)[-N, ], type = "l", lty = 1,
        ylim = c(-8, 8))


plot(S[sample.int(2000, 1),1,-N],
     type = "o", cex = 0.75, pch = 16, ylim = c(0, 1))
