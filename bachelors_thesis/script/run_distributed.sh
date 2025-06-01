#!/bin/bash

################################ COMMANDS ################################

NAVIGATOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_1000000_4_navigator.json \
 --logging-path ../data/logging/logging_distributed_1000000_4_navigator.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 0 --auxiliary-regions 0-3"

PREDICTOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_1000000_4_predictor.json \
 --logging-path ../data/logging/logging_distributed_1000000_4_predictor.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 1 --auxiliary-regions 2-3"

PATHFINDER_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_1000000_4_pathfinder.json \
 --logging-path ../data/logging/logging_distributed_1000000_4_pathfinder.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 2 --auxiliary-regions 1-2"

FORECASTER_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_1000000_4_forecaster.json \
 --logging-path ../data/logging/logging_distributed_1000000_4_forecaster.log \
 --socket-address tcp://128.131.58.90:5555 \
 --region 3 --auxiliary-regions 1-3"

NOVA_COMMAND="python cloud_server.py \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_all_4.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_1000000_4_nova.json \
 --logging-path ../data/logging/logging_distributed_1000000_4_nova.log \
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

ssh forecaster "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $FORECASTER_COMMAND'"

ssh nova "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $NOVA_COMMAND'"

# ALL dataset left out, because the kernel kills the Python process due to out-of-memory errors
