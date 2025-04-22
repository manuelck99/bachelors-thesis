import logging
import math
import time

import faiss
import numpy as np

from config import K, DIMENSION, NUMBER_OF_THREADS, SIMILARITY_THRESHOLD
from vehicle_record import VehicleRecord, VehicleRecordCluster

logger = logging.getLogger(__name__)


def top_k_search(features: list[np.ndarray],
                 features_ids: list[int],
                 k: int,
                 dimension: int,
                 number_of_threads: int) -> np.ndarray:
    logger.debug("Creating vector database")
    db_features = np.vstack(features)

    logger.debug("Normalizing vector database")
    faiss.normalize_L2(db_features)

    use_gpu = faiss.get_num_gpus() > 0

    logger.debug("Building FAISS index")
    features_index = faiss.IndexFlatIP(dimension)
    if use_gpu:
        logger.debug("Using GPU index")
        features_index = faiss.index_cpu_to_all_gpus(features_index)
    features_index = faiss.IndexIDMap(features_index)

    if not use_gpu:
        logger.debug("Setting number of threads for CPU index")
        if number_of_threads > faiss.omp_get_max_threads():
            faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        else:
            faiss.omp_set_num_threads(number_of_threads)

    logger.debug("Adding vector database to FAISS index")
    features_index.add_with_ids(db_features, features_ids)

    logger.debug("Similarity searching vector database")
    _, results = features_index.search(db_features, k)

    return results


def cluster_records(records: list[VehicleRecord]) -> set[VehicleRecordCluster]:
    # Top K rough search
    vehicle_features = [record.vehicle_feature for record in records]
    vehicle_features_ids = [record.record_id for record in records]
    license_plate_features = [record.license_plate_feature for record in records if
                              record.has_license_plate()]
    license_plate_features_ids = [record.record_id for record in records if record.has_license_plate()]

    t0 = time.time_ns()
    vehicle_top_k_results = top_k_search(vehicle_features, vehicle_features_ids, K, DIMENSION,
                                         NUMBER_OF_THREADS)
    license_plate_top_k_results = top_k_search(license_plate_features, license_plate_features_ids, K,
                                               DIMENSION,
                                               NUMBER_OF_THREADS)
    t1 = time.time_ns()
    logger.info(f"Top K search time [ms]: {(t1 - t0) / 1000 / 1000}")

    # Merging rough search results for vehicle features and license plate features
    records_dict = {record.record_id: record for record in records}
    candidate_records_dict = dict()
    for top_k_ids, record_id in zip(vehicle_top_k_results, vehicle_features_ids):
        s = {records_dict[i] for i in top_k_ids if i != record_id}
        candidate_records_dict[record_id] = s

    for top_k_ids, record_id in zip(license_plate_top_k_results, license_plate_features_ids):
        s = {records_dict[i] for i in top_k_ids if i != record_id}
        candidate_records_dict[record_id] = candidate_records_dict[record_id].union(s)

    # Clustering
    t0 = time.time_ns()
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

            if top_similarity_score > SIMILARITY_THRESHOLD:
                top_similarity_cluster.add_record(record)
            else:
                cluster = VehicleRecordCluster()
                cluster.add_record(record)
    t1 = time.time_ns()
    logger.info(f"Clustering execution time [ms]: {(t1 - t0) / 1000 / 1000}")

    clusters = {record.cluster for record in records}
    return clusters
