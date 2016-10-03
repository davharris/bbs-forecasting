devtools::load_all()
library(raster)
library(tidyverse)


# 2011 version of the 2001 NLCD data
landcover = raster("~/Downloads/nlcd_2001_landcover_2011_edition_2014_10_10/nlcd_2001_landcover_2011_edition_2014_10_10.img")


latlong = get_bbs_data() %>% 
  add_env_data() %>% 
  na.omit() %>% 
  distinct(lat, long, site_id)

locations = latlong %>% select(long, lat) %>%
  SpatialPoints(proj4string = CRS("+proj=longlat")) %>%
  spTransform(crs(landcover))

class_table = attributes(landcover)$data@attributes[[1]] %>%
  filter(COUNT >0)

route_length = 39428.93 # 24.5 miles in meters

mean_coverage = function(x){
  sapply(class_table$ID, function(id)mean(x == id, na.rm = TRUE))
}

route_class_means = raster::extract(landcover, locations, buffer = route_length) %>% 
  sapply(mean_coverage) %>% 
  t() %>% 
  as_data_frame() %>% 
  magrittr::set_names(class_table$Land.Cover.Class) %>% 
  cbind(site_id = latlong$site_id, .)
  
