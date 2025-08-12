# Currus

**Currus** is a distributed **Vehicle Trajectory Reconstruction (VTR)** system.
It aims to reconstruct the trajectories of vehicles through a large city based on vehicle snapshots taken from traffic cameras dispersed through the city.

## Centralized

The following command runs the centralized version of Currus with GPUs:

```
python centralized.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_100000_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_100000_gpu_cloud.log \
 --use-gpu
```

## Distributed

### Cloud Server

The following command runs the cloud server of Currus:

```
python cloud_server.py \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_100000_4_cloud.json \
 --logging-path ../data/logging/logging_distributed_100000_4_cloud.log \
 --socket-address tcp://localhost:5555
```

### Edge Server

The following command runs the edge server of Currus:

```
python edge_server.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_100000_4_edge_0.json \
 --logging-path ../data/logging/logging_distributed_100000_4_edge_0.log \
 --socket-address tcp://localhost:5555 \
 --region 0 --auxiliary-regions 0-1
```

## TODOs

There are a number of TODOs left in the code, which future work should examine and work through.
Additionally, there are a number of improvements that would be beneficial, such as:

- Adding encryption to messages sent through **ZeroMQ**.
- Combine standalone functions into a class (merging, clustering, map-matching, etc.) to better couple them and to make configuration easier.
- Improve configuration of the different systems. Instead of a mix of `config.py` and CLI options, everything should be handled through CLI options and actual configuration files like **JSON**, **YAML**, `.env` etc. Additionally, it would be best if configuration options were passed to objects as an object, instead of functions that just access global variables or pass configuration options along.
- Refactor code to remove redundant and/or duplicated code sections.
