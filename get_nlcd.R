library(dplyr)
library(raster)
library(RPostgreSQL)

landcover = raster("~/Downloads/nlcd_2001_landcover_2011_edition_2014_10_10/nlcd_2001_landcover_2011_edition_2014_10_10.img")
database = src_sqlite("data/bbsforecasting.sqlite")


locations = tbl(database, "bbs_data") %>% 
  dplyr::select(long, lat) %>% 
  distinct() %>% 
  as.data.frame() %>%
  SpatialPoints(proj4string = CRS("+proj=longlat")) %>%
  spTransform(crs(landcover))


class_table = attributes(landcover)$data@attributes[[1]] %>%
  filter(COUNT >0)

route_length = 39428.93 # 24.5 miles in meters

route_center_classes = unlist(
  raster::extract(landcover, locations, buffer = 0, small = TRUE)
)
route_is_valid = 

cover_by_route = raster::extract(landcover, 
                                locations, 
                                buffer = route_length
)
