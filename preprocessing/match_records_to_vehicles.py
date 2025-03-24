"""

This script matches each vehicle camera record to the corresponding vehicle
and writes the records to files "records-vehicle-VEHICLE_ID.json", where
VEHICLE_ID is the ID of the vehicle.

"""

import json

data_path = "../datasets/UrbanVehicle/records"

vehicle_file_handles = dict()
vehicle_file_handles["other"] = open(
    f"{data_path}/vehicles/records-vehicle-other.json",
    mode="w",
    encoding="utf-8"
)
with open(f"{data_path}/records.json", mode="r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)

        if record["vehicle_id"] is not None:
            if record["vehicle_id"] in vehicle_file_handles:
                vehicle_file_handles[record["vehicle_id"]].write(line)
            else:
                path = f"{data_path}/vehicles/records-vehicle-{record["vehicle_id"]}.json"
                f = open(path, mode="w", encoding="utf-8")
                f.write(line)
                vehicle_file_handles[record["vehicle_id"]] = f
        else:
            vehicle_file_handles["other"].write(line)

for file in vehicle_file_handles.values():
    file.close()
