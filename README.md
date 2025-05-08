# Bachelor's Thesis

## Centralized

```
python centralized.py --records-path ../data/dataset/records/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --map-match-proj-graph --use-gpu
```

## Distributed

### Cloud Server

```
python cloud_server.py --records-path ../data/dataset/records/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region-partitioning-2.pickle \
 --map-match-proj-graph --use-gpu
```

### Edge Server

```
python edge_server.py --records-path ../data/dataset/records/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region-partitioning-2.pickle \
 --region 0 \
 --auxiliary-regions 0-1 \
 --map-match-proj-graph --use-gpu
```
