#!/bin/bash

servers=("nova" "navigator" "predictor" "pathfinder" "forecaster")

for server in ${servers[@]}
do
  echo "Connecting to $server..."
  ssh $server '
  wget -O records.json.7z.001 "https://cloud.tsinghua.edu.cn/d/3c8bf23a1d5a4be5a20e/files/?p=%2Frecords.json.7z.001&dl=1"
  wget -O records.json.7z.002 "https://cloud.tsinghua.edu.cn/d/3c8bf23a1d5a4be5a20e/files/?p=%2Frecords.json.7z.002&dl=1"
  7z x records.json.7z.001
  ' > /dev/null 2>&1
done
