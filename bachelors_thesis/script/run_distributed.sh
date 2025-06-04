#!/bin/bash

DATASETS=("annotated" "40000" "100000" "1000000")
SESSION="distributed"
REGION="4"

CONDA_ENV="bachelors-thesis"
INIT_CONDA="source /home/manuel/anaconda3/etc/profile.d/conda.sh"
ACTIVATE_CONDA_ENV="source /home/manuel/anaconda3/bin/activate $CONDA_ENV"
PRE_COMMAND="cd ./bachelors-thesis/bachelors_thesis"

for dataset in ${DATASETS[@]}
do
  while true
  do
    ssh nova "tmux has-session -t ${SESSION}" 2>/dev/null
    if [ $? -eq 0 ]
    then
      echo "Sleeping 60s..."
      sleep 60
    else
      echo "Executing dataset ${dataset} experiments..."

      NOVA_COMMAND="python cloud_server.py \
       --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
       --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
       --region-partitioning-path ../data/road_graph/region_partitioning_all_${REGION}.pickle \
       --clusters-output-path ../data/evaluation/clusters_distributed_${dataset}_${REGION}_nova.json \
       --logging-path ../data/logging/logging_distributed_${dataset}_${REGION}_nova.log \
       --socket-address tcp://0.0.0.0:5555"

      NAVIGATOR_COMMAND="python edge_server.py \
       --records-path ../data/dataset/records-${dataset}.json \
       --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
       --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
       --region-partitioning-path ../data/road_graph/region_partitioning_all_${REGION}.pickle \
       --clusters-output-path ../data/evaluation/clusters_distributed_${dataset}_${REGION}_navigator.json \
       --logging-path ../data/logging/logging_distributed_${dataset}_${REGION}_navigator.log \
       --socket-address tcp://128.131.58.90:5555 \
       --region 0 --auxiliary-regions 0-3"

      PREDICTOR_COMMAND="python edge_server.py \
       --records-path ../data/dataset/records-${dataset}.json \
       --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
       --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
       --region-partitioning-path ../data/road_graph/region_partitioning_all_${REGION}.pickle \
       --clusters-output-path ../data/evaluation/clusters_distributed_${dataset}_${REGION}_predictor.json \
       --logging-path ../data/logging/logging_distributed_${dataset}_${REGION}_predictor.log \
       --socket-address tcp://128.131.58.90:5555 \
       --region 1 --auxiliary-regions 2-3"

      PATHFINDER_COMMAND="python edge_server.py \
       --records-path ../data/dataset/records-${dataset}.json \
       --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
       --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
       --region-partitioning-path ../data/road_graph/region_partitioning_all_${REGION}.pickle \
       --clusters-output-path ../data/evaluation/clusters_distributed_${dataset}_${REGION}_pathfinder.json \
       --logging-path ../data/logging/logging_distributed_${dataset}_${REGION}_pathfinder.log \
       --socket-address tcp://128.131.58.90:5555 \
       --region 2 --auxiliary-regions 1-2"

      FORECASTER_COMMAND="python edge_server.py \
       --records-path ../data/dataset/records-${dataset}.json \
       --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
       --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
       --region-partitioning-path ../data/road_graph/region_partitioning_all_${REGION}.pickle \
       --clusters-output-path ../data/evaluation/clusters_distributed_${dataset}_${REGION}_forecaster.json \
       --logging-path ../data/logging/logging_distributed_${dataset}_${REGION}_forecaster.log \
       --socket-address tcp://128.131.58.90:5555 \
       --region 3 --auxiliary-regions 1-3"

      ssh nova "tmux new-session -d -s ${SESSION} '${INIT_CONDA} &&
        ${ACTIVATE_CONDA_ENV} &&
        ${PRE_COMMAND} &&
        ${NOVA_COMMAND}'"

      ssh navigator "tmux new-session -d -s ${SESSION} '${INIT_CONDA} &&
        ${ACTIVATE_CONDA_ENV} &&
        ${PRE_COMMAND} &&
        ${NAVIGATOR_COMMAND}'"

      ssh predictor "tmux new-session -d -s ${SESSION} '${INIT_CONDA} &&
        ${ACTIVATE_CONDA_ENV} &&
        ${PRE_COMMAND} &&
        ${PREDICTOR_COMMAND}'"

      ssh pathfinder "tmux new-session -d -s ${SESSION} '${INIT_CONDA} &&
        ${ACTIVATE_CONDA_ENV} &&
        ${PRE_COMMAND} &&
        ${PATHFINDER_COMMAND}'"

      ssh forecaster "tmux new-session -d -s ${SESSION} '${INIT_CONDA} &&
        ${ACTIVATE_CONDA_ENV} &&
        ${PRE_COMMAND} &&
        ${FORECASTER_COMMAND}'"

      break
    fi
  done
done
