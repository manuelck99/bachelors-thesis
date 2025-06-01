#!/bin/bash

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  scp ../../data/road_graph/road_graph.zip $server:road_graph.zip
  ssh $server "mv road_graph.zip ./bachelors-thesis/data/road_graph/road_graph.zip &&
  cd ./bachelors-thesis/data/road_graph &&
  unzip road_graph.zip &&
  rm -rf road_graph.zip" > /dev/null 2>&1
done
