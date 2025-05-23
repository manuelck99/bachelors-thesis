#!/bin/bash

################################ GPU COMMANDS ################################

COMMAND_ANNOTATED_GPU="python centralized.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_annotated_gpu_nova.json \
 --logging-path ../data/logging/logging_centralized_annotated_gpu_nova.log \
 --use-gpu"

COMMAND_40000_GPU="python centralized.py \
 --records-path ../data/dataset/records-40000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_40000_gpu_nova.json \
 --logging-path ../data/logging/logging_centralized_40000_gpu_nova.log \
 --use-gpu"

COMMAND_100000_GPU="python centralized.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_100000_gpu_nova.json \
 --logging-path ../data/logging/logging_centralized_100000_gpu_nova.log \
 --use-gpu"

COMMAND_1000000_GPU="python centralized.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_1000000_gpu_nova.json \
 --logging-path ../data/logging/logging_centralized_1000000_gpu_nova.log \
 --use-gpu"

COMMAND_ALL_GPU="python centralized.py \
 --records-path ../data/dataset/records-all.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_all_gpu_nova.json \
 --logging-path ../data/logging/logging_centralized_all_gpu_nova.log \
 --use-gpu"

################################ COMMANDS ################################

COMMAND_ANNOTATED="python centralized.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_annotated_nova.json \
 --logging-path ../data/logging/logging_centralized_annotated_nova.log"

COMMAND_40000="python centralized.py \
 --records-path ../data/dataset/records-40000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_40000_nova.json \
 --logging-path ../data/logging/logging_centralized_40000_nova.log"

COMMAND_100000="python centralized.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_100000_nova.json \
 --logging-path ../data/logging/logging_centralized_100000_nova.log"

COMMAND_1000000="python centralized.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_1000000_nova.json \
 --logging-path ../data/logging/logging_centralized_1000000_nova.log"

COMMAND_ALL="python centralized.py \
 --records-path ../data/dataset/records-all.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_all_nova.json \
 --logging-path ../data/logging/logging_centralized_all_nova.log"

################################ SSH ################################

CONDA_ENV="bachelors-thesis"
INIT_CONDA="source /home/manuel/anaconda3/etc/profile.d/conda.sh"
ACTIVATE_CONDA_ENV="source /home/manuel/anaconda3/bin/activate $CONDA_ENV"
SESSION_NAME="centralized"
PRE_COMMAND="cd ./bachelors-thesis/bachelors_thesis"

ssh nova "tmux new-session -d -s $SESSION_NAME '$INIT_CONDA &&
  $ACTIVATE_CONDA_ENV &&
  $PRE_COMMAND &&
  $COMMAND_ANNOTATED_GPU &&
  $COMMAND_40000_GPU &&
  $COMMAND_100000_GPU &&
  $COMMAND_1000000_GPU &&
  $COMMAND_ALL_GPU &&
  $COMMAND_ANNOTATED &&
  $COMMAND_40000 &&
  $COMMAND_100000 &&
  $COMMAND_1000000 &&
  $COMMAND_ALL'"
