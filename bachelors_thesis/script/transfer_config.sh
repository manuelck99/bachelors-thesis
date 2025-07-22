#!/bin/bash

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  scp ../config.py $server:config.py
  ssh $server "mv config.py ./bachelors-thesis/bachelors_thesis/config.py" > /dev/null 2>&1
done
