library(forecast)
library(lme4)
library(rstan)
library(mvtnorm)
library(tidyr)
library(dplyr)
devtools::load_all()

start_yr <- 1982
end_yr <- 2013
min_num_yrs <- 25
last_training_year <- 2003
richness_w_env <- get_richness_ts_env_data(start_yr, end_yr, min_num_yrs) %>%
  subset(year <=last_training_year) %>%
  na.omit()

arima_data = richness_w_env %>%
  dplyr::select(site_id, year, richness) %>%
  group_by(site_id) %>%
  tidyr::spread(key = year, value = richness) %>%
  t() %>%
  tbl_df()
colnames(arima_data) = arima_data[1, ]
arima_data = arima_data[-1, ]

naive_sigmas = sapply(arima_data, function(x) sd(diff(x), na.rm = TRUE))

diffs = sapply(arima_data, function(x) diff(x))
diff_var = apply(diffs, 2, var, na.rm = TRUE)

lmer_model = lmer(
  richness ~ (1|site_id),
  data = richness_w_env
)

resid_var = data_frame(site_id = richness_w_env$site_id, resid = resid(lmer_model)) %>%
  group_by(site_id) %>%
  summarize(var(resid)) %>%
  magrittr::extract2("var(resid)")


# choose Y1 ---------------------------------------------------------------

site_id = sample(names(which(colSums(is.na(arima_data))==0)), 1)
y1 = arima_data[[site_id]]
plot(y1)

# Priors ------------------------------------------------------------------

# Empirical prior distribution on mu
means_prior = c(
  fixef(lmer_model),
  sd(ranef(lmer_model)$site_id[[1]])
)

# Prior on fn_sd
fn_scale_prior = MASS::fitdistr(sqrt(resid_var), "gamma")$estimate
fn_scale_prior[1] = fn_scale_prior[1] / 2 # Flatten the prior
fn_scale_prior[2] = fn_scale_prior[2] / 4 # Stretch the prior to the right because we haven't observed full range of variation
 curve(dgamma(x, fn_scale_prior[1], fn_scale_prior[2]), to = 2 * sd(richness_w_env$richness), xlab = "richness fn_sd")
 abline(v = sqrt(mean(resid_var)))

# prior distribution on nugget_sd
nugget_prior = c(0, sd(y1) / 2)
 curve(dnorm(x, nugget_prior[1], nugget_prior[2]), to = sqrt(2 * var(y1)), xlab = "richness nugget")
 abline(v = sqrt(var(y1)/2))

# Prior distribution on the lengthscale
lengthscale_prior = c(4, 1/5)
 curve(dgamma(x, lengthscale_prior[1], lengthscale_prior[2]), to = 300, xlab = "years")
 abline(v = 1)
 abline(v = 20)


# GP ----------------------------------------------------------------------

N1 = length(y1)
N2 = 10
N = N1 + N2
x = 1:N

d = as.matrix(dist(x))

data = list(N1 = N1, N2 = N2, x = x, y1 = y1, d = d, means_prior = means_prior,
            fn_scale_prior = fn_scale_prior, nugget_prior = nugget_prior,
            lengthscale_prior = lengthscale_prior,
            full_sd = sd(unlist(arima_data), na.rm = TRUE))

m = stan_model(
  "gp/gp.stan"
)

model = sampling(m, data = data, control = list(adapt_delta = 0.95),
                 chains = 2)



matplot(xlim = c(1, N), seq(N1+1, N), t(rstan::extract(model, "y2")[[1]]), col = "#00000010", lty = 1, pch = 16, type = "n",
        xlab = "years since start", ylab = "richness")
points(y1)
matlines(seq(N1+1, N), t(apply(t(rstan::extract(model, "y2")[[1]]), 1, quantile, c(.025, .975))), col = "maroon", lwd = 2, lty = 3)
rug(rstan::extract(model, "mu")[[1]], side = 2, col = "#00000010")
lines(seq(N1+1, N), colMeans(rstan::extract(model, "y2")[[1]]), col = "forestgreen", lwd = 4)
abline(h = range(y1), col = "#00000020")
abline(h = mean(y1), col = "#00000040")

par(mfrow = c(2, 2))

plot(density(rstan::extract(model, "mu")[[1]], adj = .1), col = 2, main = "mean richness")
curve(dnorm(x, means_prior[1], means_prior[2]), xlab = "years", add = TRUE, from = 0)

plot(density(rstan::extract(model, "fn_sd")[[1]], adj = .1), col = 2, main = "fn_sd")
curve(dgamma(x, fn_scale_prior[1], fn_scale_prior[2]), add = TRUE)
abline(v = sd(y1))

plot(density(rstan::extract(model, "lengthscale")[[1]], adj = .1, n = 1000), col = 2, main = "lengthscale years")
curve(dgamma(x, lengthscale_prior[1], lengthscale_prior[2]), xlab = "years", add = TRUE, from = 0, n = 1000)

plot(density(rstan::extract(model, "nugget_sd")[[1]], adj = .1), col = 2, main = "nugget_sd")
curve(dnorm(x, nugget_prior[1], nugget_prior[2]), xlab = "years", add = TRUE, from = 0)
abline(v = sd(y1))

par(mfrow = c(1, 1))

library(mvtnorm)

par(mfrow = c(3, 3))
replicate(9,{
  i = sample.int(dim(rstan::extract(model, "Sigma")[[1]])[1], 1)
  matplot(t(rmvnorm(5, sigma = rstan::extract(model, "Sigma")[[1]][i,,])), type = "l", lty = 1, ylim = c(-3, 3))
})
par(mfrow = c(1, 1))
