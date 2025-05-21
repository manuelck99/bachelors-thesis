#!/bin/bash

CONDA_ENV="bachelors-thesis"
INIT_CONDA="source /home/manuel/anaconda3/etc/profile.d/conda.sh"
ACTIVATE_CONDA_ENV="source /home/manuel/anaconda3/bin/activate $CONDA_ENV"


############################################## PARAMETERS ##############################################

SESSION_NAME="distributed_annotated_gpu_2"

PRE_COMMAND="cd ./bachelors-thesis/bachelors_thesis"

NOVA_COMMAND="python cloud_server.py \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_annotated_2.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_gpu_2_nova.json \
 --socket-address tcp://0.0.0.0:5555 \
 --use-gpu"

NAVIGATOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_2_annotated.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_gpu_2_navigator.json \
 --socket-address tcp://128.131.58.90:5555 \
 --region 0 --auxiliary-regions 0-1"

PREDICTOR_COMMAND="python edge_server.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --region-partitioning-path ../data/road_graph/region_partitioning_2_annotated.pickle \
 --clusters-output-path ../data/evaluation/clusters_distributed_annotated_gpu_2_predictor.json \
 --socket-address tcp://128.131.58.90:5555 \
 --region 1"

#FORECASTER_COMMAND=""

#PATHFINDER_COMMAND=""

############################################## PARAMETERS ##############################################

echo "Starting nova server inside tmux session..."
ssh nova "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA && $ACTIVATE_CONDA_ENV && $PRE_COMMAND && $NOVA_COMMAND; bash'"
echo "Started nova server"

echo "Starting navigator server inside tmux session..."
ssh navigator "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA && $ACTIVATE_CONDA_ENV && $PRE_COMMAND && $NAVIGATOR_COMMAND; bash'"
echo "Started navigator server"

echo "Starting predictor server inside tmux session..."
ssh predictor "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA && $ACTIVATE_CONDA_ENV && $PRE_COMMAND && $PREDICTOR_COMMAND; bash'"
echo "Started predictor server"

#echo "Starting forecaster server inside tmux session..."
#ssh forecaster "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA && $ACTIVATE_CONDA_ENV && $PRE_COMMAND && $FORECASTER_COMMAND; bash'"
#echo "Started forecaster server"

#echo "Starting pathfinder server inside tmux session..."
#ssh pathfinder "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA && $ACTIVATE_CONDA_ENV && $PRE_COMMAND && $PATHFINDER_COMMAND; bash'"
#echo "Started pathfinder server"
