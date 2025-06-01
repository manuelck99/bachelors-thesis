#!/bin/bash

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  ssh $server "cd ./bachelors-thesis/data/road_graph &&
    rm -rf region_partitioning* road_graph*" > /dev/null 2>&1
done
