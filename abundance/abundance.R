library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)
library(purrr)
library(rstan)
devtools::load_all()
if (!exists("x")) {
  x = get_bbs_data()
}
set.seed(1)


spid = 6870

d = x %>%
  filter(species_id == spid, year == 2010) %>%
  dplyr::select(site_id, abundance) %>%
  full_join(
    distinct(dplyr::select(x, site_id, lat, long), site_id, .keep_all = TRUE), 
    by = "site_id") %>%
  replace_na(list(abundance = 0))

locations = distinct(d, long, lat)
centers = kmeans(locations, 20, nstart = 50)$centers
distances = raster::pointDistance(cbind(d$long, d$lat), centers, lonlat = TRUE)
kernel = exp(-1/2 * distances^2 / 1E6^2)
kernel = kernel[ , -caret::findCorrelation(cor(kernel), cutoff = .8)]


ggplot(NULL) + 
  geom_point(aes(y = d$lat, x = d$long, color = kernel[,2])) + 
  scale_color_viridis() + 
  coord_equal() +
  geom_point(aes(x = centers[,1], y = centers[,2]), color = "red")

data = list(
  y = d$abundance,
  x = kernel
)
data$N = nrow(data$x)
data$P = ncol(data$x)

compiled = stan_model("abundance.stan")

m = sampling(compiled, data = data, verbose = TRUE, chains = 1)
extracted = extract(m)


data_frame(
  p_zero = colMeans(plogis(extracted$logit_p_zero)),
  lambda = colMeans(extracted$lambda),
  y = data$y
) %>% 
  ggplot(aes(x = lambda, y = y)) + 
  geom_point(aes(alpha = 1 - p_zero)) +
  geom_abline(intercept = 0, slope = 1)



data_frame(
  p_not_blocked = 1 - colMeans(plogis(extracted$logit_p_zero)),
  abundance_if_occur = colMeans(extracted$predicted_lambda),
  x = d$long,
  y = d$lat
) %>% ggplot(aes(x = x, y = y, color = abundance_if_occur, alpha = p_not_blocked)) +
  geom_point() +
  scale_color_viridis(option = "A", direction = -1) + 
  theme_bw() + 
  coord_equal()
  
