#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
source ~/anaconda3/bin/activate bachelors-thesis

echo "Attaching IDs..."
python attach_ids_to_records.py -i ../../data/dataset/records/records.json -o ../../data/dataset/records/records-with-ids.json
echo "Finished attaching IDs"

echo "Sorting records..."
python sort_records.py -i ../../data/dataset/records/records-with-ids.json -o ../../data/dataset/records/records-sorted.json
echo "Finished sorting records"

echo "Generating datasets..."
python generate_datasets.py -i ../../data/dataset/records/records-sorted.json -o ../../data/dataset/records --sizes 40000 100000 1000000
echo "Finished generating datasets"
