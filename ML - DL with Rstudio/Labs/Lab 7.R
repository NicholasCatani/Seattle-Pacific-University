### Libraries

library(nycflights13)
library(dplyr)
library(igraph)

### Dataset

df <- nycflights13::flights

### Computations

unique_routes <- df %>%
  distinct(origin, dest) %>%
  mutate(weight = 1)
print(nrow(unique_routes))



unique_origins <- unique(df$origin)
unique_dests <- unique(df$dest)
unique_locations <- unique(c(unique_origins, unique_dests))
print(length(unique_locations))



routes <- df %>%
  select(origin, dest) %>%
  distinct()
g <- graph_from_data_frame(routes, directed = TRUE)
centrality <- degree(g, mode = "all")
print(max(centrality))



betweenness_data <- betweenness(g, directed = TRUE)
print(max(betweenness_data))



closeness_data <- closeness(g, mode = "all")
print(round(max(closeness_data)))


