#!/bin/bash

CONDA_ENV="bachelors-thesis"
INIT_CONDA="source /home/manuel/anaconda3/etc/profile.d/conda.sh"
ACTIVATE_CONDA="source /home/manuel/anaconda3/bin/activate"
PRE_COMMAND="cd ./bachelors-thesis"

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  ssh $server "$INIT_CONDA &&
    $ACTIVATE_CONDA base &&
    $PRE_COMMAND &&
    git pull &&
    conda env remove -n $CONDA_ENV -y ;
    conda clean --all -y &&
    conda env create -f environment.yml -y &&
    $ACTIVATE_CONDA $CONDA_ENV &&
    pip install mappymatch" > /dev/null 2>&1
done
