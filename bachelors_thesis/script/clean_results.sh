#!/bin/bash

PRE_COMMAND="cd ./bachelors-thesis/data"

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  ssh $server "$PRE_COMMAND &&
    cd ./logging &&
    rm -rf logging* &&
    cd .. &&
    cd ./evaluation &&
    rm -rf clusters*" > /dev/null 2>&1
done
