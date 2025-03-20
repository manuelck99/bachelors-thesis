import logging
import math
import time

import config as cfg
from clustering import search
from util import load_records
from vehicle_record import VehicleRecordCluster

logger = logging.getLogger(__name__)

data_path = "../datasets/UrbanVehicle/records/vehicles"


def run():
    """
    Load records, assign each a numerical id, turn them into VehicleRecords and also keep track of the ids,
    making sure that you have two lists: one with VehicleRecords and other with ids all in corresponding order.
    Don't forget you have to separate VehicleRecords with license plates and those without.

    Feed them into the clustering search and it will return for each id, which VehicleRecords are most similar.
    Filter the record itself out.

    Merge the most similar VehicleRecords for each VehicleRecord

    For each record look through their most similar records and get the clusters those records belong to.
    Then calculate the visual similarity between the record and each cluster's centroid.
    Choose the cluster with the highest similarity and if its score is higher than the threshold add the record
    to the cluster (recalculate centroid), otherwise create a new cluster and add the record to it.
    """

    records = load_records([f"{data_path}/records-vehicle-{i}.json" for i in range(cfg.NUMBER_OF_LABELLED_VEHICLES)])
    logger.debug(f"Number of records: {len(records)}")

    vehicle_features = [record.vehicle_feature for record in records]
    vehicle_features_ids = [record.record_id for record in records]
    license_plate_features = [record.license_plate_feature for record in records if
                              record.has_license_plate()]
    license_plate_features_ids = [record.record_id for record in records if record.has_license_plate()]

    t0 = time.time_ns()
    vehicle_top_k_results = search(vehicle_features, vehicle_features_ids, cfg.K, cfg.DIMENSION, cfg.NUMBER_OF_THREADS)
    license_plate_top_k_results = search(license_plate_features, license_plate_features_ids, cfg.K, cfg.DIMENSION,
                                         cfg.NUMBER_OF_THREADS)
    t1 = time.time_ns()
    logger.debug(f"Similarity search time [ms]: {(t1 - t0) / 1000 / 1000}")

    records_dict = {record.record_id: record for record in records}
    candidate_records_dict = dict()
    for top_k_ids, record_id in zip(vehicle_top_k_results, vehicle_features_ids):
        s = {records_dict[i] for i in top_k_ids if i != record_id}
        candidate_records_dict[record_id] = s

    for top_k_ids, record_id in zip(license_plate_top_k_results, license_plate_features_ids):
        s = {records_dict[i] for i in top_k_ids if i != record_id}
        candidate_records_dict[record_id] = candidate_records_dict[record_id].union(s)

    for record_id, candidate_records in candidate_records_dict.items():
        record = records_dict[record_id]
        candidate_clusters = {record.cluster for record in candidate_records if record.has_assigned_cluster()}

        if len(candidate_clusters) == 0:
            cluster = VehicleRecordCluster()
            cluster.add_record(record)
        else:
            top_similarity_score = -math.inf
            top_similarity_cluster = None

            for cluster in candidate_clusters:
                similarity_score = cluster.calculate_similarity_to_record(record)
                if similarity_score > top_similarity_score:
                    top_similarity_score = similarity_score
                    top_similarity_cluster = cluster

            if top_similarity_score > cfg.SIMILARITY_THRESHOLD:
                top_similarity_cluster.add_record(record)
            else:
                cluster = VehicleRecordCluster()
                cluster.add_record(record)


if __name__ == "__main__":
    run()
