"""

This script matches each vehicle camera record to the camera it was taken from
and writes the records to files "records-camera-CAMERA_ID.json", where
CAMERA_ID is the ID of the camera.

"""

import json
import os

dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/dataset/records")

if __name__ == "__main__":
    camera_file_handles = dict()
    with open(f"{dataset_path}/records.json", mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["camera_id"] in camera_file_handles:
                camera_file_handles[record["camera_id"]].write(line)
            else:
                path = f"{dataset_path}/cameras/records-camera-{record['camera_id']}.json"
                f = open(path, mode="w", encoding="utf-8")
                f.write(line)
                camera_file_handles[record["camera_id"]] = f

    for file in camera_file_handles.values():
        file.close()
