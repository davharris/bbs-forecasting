library(viridis)
library(tidyverse)
library(rstan)
devtools::load_all()
if (!exists("x")) {
  x = get_bbs_data()
}
set.seed(1)


species_data = get_species_data() %>% 
  select(-species_id) %>% 
  rename(species_id = aou)

d = x %>% 
  left_join(species_data) %>% 
  select(site_id, english_common_name, abundance, lat, long, year) %>% 
  filter(year == 2010) %>% 
  filter(grepl("Woodpecker", .$english_common_name)) %>% 
  spread(english_common_name, abundance, fill = 0)

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

y = d %>% select(-site_id, -lat, -long, -year) %>% as.matrix() %>% t()
y = y[rowSums(y > 0) > 100, ]

data = list(
  y = y,
  x = kernel
)
data$N = nrow(data$x)
data$P = ncol(data$x)
data$N_species = nrow(data$y)

compiled = stan_model("abundance/abundance.stan")

m = sampling(compiled, data = data, verbose = TRUE, chains = 1)
extracted = extract(m)


i = 5
data_frame(
  p_not_blocked = 1 - colMeans(plogis(extracted$alpha_zero[,i] + extracted$x_beta_zero[,,i])),
  abundance_if_occur = colMeans(exp(extracted$alpha_count[,i] +  extracted$x_beta_count[,,i] + extracted$sigma[,i] * rnorm(prod(dim(extracted$x_beta_count[,,i]))))),
  x = d$long,
  y = d$lat
) %>% ggplot(aes(x = x, y = y, color = abundance_if_occur, alpha = p_not_blocked)) +
  geom_point() +
  scale_color_viridis(option = "A", direction = -1) + 
  theme_bw() + 
  coord_equal()
  
