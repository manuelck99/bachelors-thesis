import math
import os.path
from collections import defaultdict

from config import NUMBER_OF_LABELLED_VEHICLES
from vehicle_record import VehicleRecord, VehicleRecordCluster

Precision = float
Recall = float
F1Score = float
Expansion = float

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/dataset/records/vehicles")


def yu_ao_yan_evaluation(records: list[VehicleRecord], clusters: set[VehicleRecordCluster]) -> tuple[
    Precision, Recall, F1Score, Expansion]:
    vehicle_records_count = defaultdict(int)
    for record in records:
        if record.is_annotated():
            vehicle_records_count[record.vehicle_id] += 1

    # TODO: Replace NUMBER_OF_LABELLED_VEHICLES with len(vehicle_records_count) and refactor where necessary
    precision = 0.0
    recall = 0.0
    for vehicle_id in range(NUMBER_OF_LABELLED_VEHICLES):
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                    cluster_of_vehicle)
        precision += number_of_records_of_vehicle_in_cluster / len(cluster_of_vehicle.records)
        recall += number_of_records_of_vehicle_in_cluster / vehicle_records_count[vehicle_id]

    precision /= NUMBER_OF_LABELLED_VEHICLES
    recall /= NUMBER_OF_LABELLED_VEHICLES
    f1_score = (precision * recall) / (precision + recall)

    expansion = 0.0
    for vehicle_id in range(NUMBER_OF_LABELLED_VEHICLES):
        for cluster in clusters:
            number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                        cluster)
            if number_of_records_of_vehicle_in_cluster != 0:
                expansion += 1.0
    expansion /= NUMBER_OF_LABELLED_VEHICLES

    return precision, recall, f1_score, expansion


def find_cluster_of_vehicle(vehicle_id: int, clusters: set[VehicleRecordCluster]) -> VehicleRecordCluster | None:
    max_number_of_records = -math.inf
    max_number_of_records_cluster = None
    for cluster in clusters:
        number_of_records = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id, cluster)
        if number_of_records > max_number_of_records:
            max_number_of_records = number_of_records
            max_number_of_records_cluster = cluster
    return max_number_of_records_cluster


def calculate_number_of_records_of_vehicle_in_cluster(vehicle_id: int, cluster: VehicleRecordCluster) -> int:
    count = 0
    for record in cluster.records.values():
        if record.vehicle_id is not None and record.vehicle_id == vehicle_id:
            count += 1
    return count
