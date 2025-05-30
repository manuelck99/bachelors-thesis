#!/bin/bash

################################ GPU COMMANDS ################################

COMMAND_ANNOTATED_GPU="python centralized.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_annotated_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_annotated_gpu_cloud.log \
 --use-gpu"

COMMAND_40000_GPU="python centralized.py \
 --records-path ../data/dataset/records-40000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_40000_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_40000_gpu_cloud.log \
 --use-gpu"

COMMAND_100000_GPU="python centralized.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_100000_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_100000_gpu_cloud.log \
 --use-gpu"

COMMAND_1000000_GPU="python centralized.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_1000000_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_1000000_gpu_cloud.log \
 --use-gpu"

COMMAND_ALL_GPU="python centralized.py \
 --records-path ../data/dataset/records-all.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_all_gpu_cloud.json \
 --logging-path ../data/logging/logging_centralized_all_gpu_cloud.log \
 --use-gpu"

################################ COMMANDS ################################

COMMAND_ANNOTATED="python centralized.py \
 --records-path ../data/dataset/records-annotated.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_annotated_cloud.json \
 --logging-path ../data/logging/logging_centralized_annotated_cloud.log"

COMMAND_40000="python centralized.py \
 --records-path ../data/dataset/records-40000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_40000_cloud.json \
 --logging-path ../data/logging/logging_centralized_40000_cloud.log"

COMMAND_100000="python centralized.py \
 --records-path ../data/dataset/records-100000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_100000_cloud.json \
 --logging-path ../data/logging/logging_centralized_100000_cloud.log"

COMMAND_1000000="python centralized.py \
 --records-path ../data/dataset/records-1000000.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_1000000_cloud.json \
 --logging-path ../data/logging/logging_centralized_1000000_cloud.log"

COMMAND_ALL="python centralized.py \
 --records-path ../data/dataset/records-all.json \
 --road-graph-path ../data/road_graph/road_graph_ox_nsl_sim_sc.pickle \
 --cameras-info-path ../data/road_graph/road_graph_ox_nsl_sim_sc_cameras.pickle \
 --clusters-output-path ../data/evaluation/clusters_centralized_all_cloud.json \
 --logging-path ../data/logging/logging_centralized_all_cloud.log"

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
  $COMMAND_ANNOTATED &&
  $COMMAND_40000 &&
  $COMMAND_100000 &&
  $COMMAND_1000000'"

# ALL dataset left out, because the kernel kills the Python process due to out-of-memory errors
