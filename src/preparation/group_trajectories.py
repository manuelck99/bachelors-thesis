import json
from collections import defaultdict


data_path = "../../datasets/UrbanVehicle"

vehicle_trajectories_count = defaultdict(int)
with open(f"{data_path}/trajectories.json", "r") as file:
    for line in file:
        trajectory = json.loads(line)
        vehicle_trajectories_count[trajectory["vehicle_id"]] += 1

multiple_trajectories_vehicles = list()
for vehicle_id, count in vehicle_trajectories_count.items():
    if count > 1:
        multiple_trajectories_vehicles.append(vehicle_id)

if multiple_trajectories_vehicles:
    print("Following vehicles have multiple trajectories:")
    print(multiple_trajectories_vehicles)
else:
    print("No vehicle has multiple trajectories")