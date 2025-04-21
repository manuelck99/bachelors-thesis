"""

This script matches each vehicle camera record to the camera it was taken from
and writes the records to files "records-camera-CAMERA_ID.json", where
CAMERA_ID is the ID of the camera.

"""

import json
import os.path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input-path",
        type=str,
        required=True,
        help="Path to the input records file"
    )
    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=True,
        help="Path to the output cameras folder"
    )
    args = parser.parse_args()

    camera_file_handles = dict()
    with open(args.input_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            if record["camera_id"] in camera_file_handles:
                camera_file_handles[record["camera_id"]].write(line)
            else:
                path = os.path.join(args.output_path, f"records-camera-{record['camera_id']}.json")
                f = open(path, mode="w", encoding="utf-8")
                f.write(line)
                camera_file_handles[record["camera_id"]] = f

    for file in camera_file_handles.values():
        file.close()
