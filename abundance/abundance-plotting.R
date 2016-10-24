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








Z = as.data.frame(read.csv("Z.csv", header = FALSE))
Z_variance = as.data.frame(read.csv("Z_variance.csv", header = FALSE))
log_y_hat = read.csv("log_y_hat.csv", header = FALSE)
p_inflated = plogis(as.matrix(read.csv("p_zero.csv", header = FALSE)))



#i = sample.int(ncol(log_y_hat), 1)
rownames(y)[i]

ggplot(d) + 
  geom_polygon(aes_(~long, ~lat, group = ~group), 
    data = map_data("world", xlim = range(d$long), ylim = range(d$lat)), 
    fill = NA, color = "gray90", inherit.aes = FALSE)  + 
  geom_point(aes(y = lat, x = long, color = exp(log_y_hat[,i]), alpha = 1 - p_inflated[,i])) + 
  scale_color_viridis(
    option = "A", 
    direction = -1, 
    limits = c(0, exp(max(log_y_hat[[i]])))
  ) + 
  scale_alpha_continuous(range = c(0, 1), limits = c(0, 1)) + 
  coord_equal(xlim = range(d$long), ylim = range(d$lat)) + 
  ggtitle(rownames(y)[i]) + 
  theme_bw()




plot(qlogis(p_inflated[, i]), log_y_hat[,i])


plot(exp(log_y_hat[,i]), asp = 1, col = alpha("black", 1 - p_inflated[,i]), y[i, ])
abline(0,1)




j = ifelse(j==ncol(Z), 1, j+1)
ggplot(d) + 
  geom_point(aes(y = lat, x = long, color = predict(prcomp(Z))[,j])) + 
  scale_color_gradient2() + 
  coord_equal() + 
  theme_bw()

ggplot(d) + 
  geom_point(aes(y = lat, x = long, color = Z_variance[[1]])) + 
  coord_equal() + 
  theme_bw() +
  scale_color_viridis()


plot(
  unlist(exp(log_y_hat)), jitter(c(t(y))), log = "xy", pch = ".", col = alpha("red", 1 - unlist(p_inflated)), cex = 2,
  xlab = "predicted",
  ylab = "observed",
  ylim = c(.99, max(y)))
abline(0,1)
