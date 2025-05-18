# Bachelor's Thesis

## Centralized

```
python centralized.py --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras_annotated.pickle \
 --clusters-output-path ../data/evaluation/clusters_gt_annotated_centralized \
 --use-gpu
```

## Distributed

### Cloud Server

```
python cloud_server.py --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras_annotated.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_2_annotated.pickle \
 --clusters-input-path ../data/evaluation/clusters_gt_annotated_centralized \
 --use-gpu
```

### Edge Server

```
python edge_server.py --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras_annotated.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_2_annotated.pickle \
 --region 0 --auxiliary-regions 0-1 \
 --use-gpu
```
