import logging
import math
import time

import faiss
import numpy as np

from config import K, DIMENSION, NUMBER_OF_THREADS, SIMILARITY_THRESHOLD, WEIGHT_VEHICLE_SIMILARITY, \
    WEIGHT_LICENSE_PLATE_SIMILARITY
from vehicle_record import VehicleRecord, VehicleRecordCluster

logger = logging.getLogger(__name__)


def top_k_search(features: list[np.ndarray],
                 *,
                 k: int,
                 dimension: int,
                 number_of_threads: int,
                 use_gpu=False) -> np.ndarray:
    db_features = np.vstack(features)
    faiss.normalize_L2(db_features)

    features_index = faiss.IndexFlatIP(dimension)
    if use_gpu:
        features_index = faiss.index_cpu_to_all_gpus(features_index)

    if not use_gpu:
        if number_of_threads > faiss.omp_get_max_threads():
            faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        else:
            faiss.omp_set_num_threads(number_of_threads)

    features_index.add(db_features)
    _, results = features_index.search(db_features, k)
    return results


def cluster_records(records: list[VehicleRecord], *, use_gpu=False) -> set[VehicleRecordCluster]:
    # Top K rough search
    vehicle_features = [record.get_vehicle_feature() for record in records]
    vehicle_features_ids = {i: record.get_record_id() for i, record in enumerate(records)}
    license_plate_features = [record.get_license_plate_feature() for record in
                              filter(lambda r: r.has_license_plate(), records)]
    license_plate_features_ids = {i: record.get_record_id() for i, record in
                                  enumerate(filter(lambda r: r.has_license_plate(), records))}

    t0 = time.time_ns()
    vehicle_top_k_results = top_k_search(vehicle_features,
                                         k=K,
                                         dimension=DIMENSION,
                                         number_of_threads=NUMBER_OF_THREADS,
                                         use_gpu=use_gpu)
    license_plate_top_k_results = top_k_search(license_plate_features,
                                               k=K,
                                               dimension=DIMENSION,
                                               number_of_threads=NUMBER_OF_THREADS,
                                               use_gpu=use_gpu)
    t1 = time.time_ns()
    logger.info(f"Top K search time [ms]: {(t1 - t0) / 1000 / 1000}")

    # Merging rough search results for vehicle features and license plate features
    records_dict = {record.get_record_id(): record for record in records}
    candidate_records_dict = dict()
    for i, top_k_ids in enumerate(vehicle_top_k_results):
        record_id = vehicle_features_ids[i]
        s = {records_dict[vehicle_features_ids[top_k_id]] for top_k_id in top_k_ids if top_k_id != i}
        candidate_records_dict[record_id] = s

    for i, top_k_ids in enumerate(license_plate_top_k_results):
        record_id = license_plate_features_ids[i]
        s = {records_dict[license_plate_features_ids[top_k_id]] for top_k_id in top_k_ids if top_k_id != i}
        candidate_records_dict[record_id] = candidate_records_dict[record_id].union(s)

    # Clustering
    t0 = time.time_ns()
    for record_id, candidate_records in candidate_records_dict.items():
        record = records_dict[record_id]
        candidate_clusters = {record.get_cluster() for record in candidate_records if record.has_assigned_cluster()}

        if len(candidate_clusters) == 0:
            cluster = VehicleRecordCluster(dimension=DIMENSION,
                                           weight_vehicle_similarity=WEIGHT_VEHICLE_SIMILARITY,
                                           weight_license_plate_similarity=WEIGHT_LICENSE_PLATE_SIMILARITY)
            cluster.add_record(record)
        else:
            top_similarity_score = -math.inf
            top_similarity_cluster = None

            for cluster in candidate_clusters:
                similarity_score = cluster.calculate_similarity_to_record(record)
                if similarity_score > top_similarity_score:
                    top_similarity_score = similarity_score
                    top_similarity_cluster = cluster

            if top_similarity_score > SIMILARITY_THRESHOLD:
                top_similarity_cluster.add_record(record)
            else:
                cluster = VehicleRecordCluster(dimension=DIMENSION,
                                               weight_vehicle_similarity=WEIGHT_VEHICLE_SIMILARITY,
                                               weight_license_plate_similarity=WEIGHT_LICENSE_PLATE_SIMILARITY)
                cluster.add_record(record)
    t1 = time.time_ns()
    logger.info(f"Clustering execution time [ms]: {(t1 - t0) / 1000 / 1000}")

    clusters = {record.get_cluster() for record in records}
    return clusters
