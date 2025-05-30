#!/bin/bash

################################ COMMANDS ################################

NAVIGATOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_3.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_3_edge.json \
 --logging-path ../data/logging/logging_distributed_annotated_3_edge.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 0 --auxiliary-regions 0-2"

PREDICTOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_3.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_3_edge.json \
 --logging-path ../data/logging/logging_distributed_annotated_3_edge.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 1 --auxiliary-regions 1-2"

PATHFINDER_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_3.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_3_edge.json \
 --logging-path ../data/logging/logging_distributed_annotated_3_edge.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 2"

FORECASTER_COMMAND=""

NOVA_COMMAND="python cloud_server.py \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_3.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_3_cloud.json \
 --logging-path ../data/logging/logging_distributed_annotated_3_cloud.log \
 --socket-address tcp://0.0.0.0:5555"

################################ SSH ################################

CONDA_ENV="bachelors-thesis"
INIT_CONDA="source /home/manuel/anaconda3/etc/profile.d/conda.sh"
ACTIVATE_CONDA_ENV="source /home/manuel/anaconda3/bin/activate $CONDA_ENV"
SESSION_NAME="distributed"
PRE_COMMAND="cd ./bachelors-thesis/bachelors_thesis"

ssh navigator "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $NAVIGATOR_COMMAND'"

ssh predictor "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $PREDICTOR_COMMAND'"

ssh pathfinder "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $PATHFINDER_COMMAND'"

ssh nova "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $NOVA_COMMAND'"

# ALL dataset left out, because the kernel kills the Python process due to out-of-memory errors
