#!/bin/bash

PRE_COMMAND="cd ./bachelors-thesis"

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  ssh $server "$PRE_COMMAND &&
    git pull" > /dev/null 2>&1
done
