#!/apps/R/3.2.0/bin/Rscript

devtools::load_all()
library(parallel)
library(tidyverse)
library(mistnet2)

dir.create("sdm_predictions", showWarnings = FALSE)

start_yr = 1982
end_yr = 2013
min_num_yrs = 25
last_train_year = 2003

# Munge -------------------------------------------------------------------
d = get_pop_ts_env_data(start_yr, end_yr, min_num_yrs) 
d = d %>% 
  add_ranefs(last_training_year = last_train_year) %>% 
  filter(!is.na(abundance)) %>% 
  mutate(present = abundance > 0) %>% 
  select(-abundance) %>% 
  spread(key = species_id, value = present, fill = 0)

is_y = grepl("^[0-9]+$", colnames(d))

train_x = d[d$year <= last_train_year, ] %>% 
  select(bio2, bio5, bio15, ndvi_sum, observer_effect) %>% 
  as.matrix()
train_y = as.matrix(d[d$year <= last_train_year, is_y])


net = mistnet(
  train_x, 
  train_y, 
  n_z = 2,
  layers = list(
    mistnet2::layer(
      activator = elu_activator,
      n_nodes = 10,
      weight_prior = make_distribution("NO", mu = 0, sigma = 1)
    ),
    mistnet2::layer(
      activator = sigmoid_activator,
      n_nodes = ncol(train_y),
      weight_prior = make_distribution("NO", mu = 0, sigma = 1)
    )
  ),
  error_distribution = make_distribution("BI", bd = 1),
  fit = FALSE
)

