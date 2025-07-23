# Bachelor's Thesis

## Centralized

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

```
python edge_server.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_100000_4_edge.json \
 --logging-path ../data/logging/logging_distributed_100000_4_edge.log \
 --socket-address tcp://localhost:5555 \
 --region 0 --auxiliary-regions 0-1
```
