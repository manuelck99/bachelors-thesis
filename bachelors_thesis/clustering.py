import logging
import math

import faiss
import numpy as np

import bachelors_thesis.config as cfg
import bachelors_thesis.vehicle_record as vr

logger = logging.getLogger(__name__)


def top_k_search(features,
                 features_ids,
                 k,
                 dimension,
                 number_of_threads):
    logger.debug("Creating vector database")
    db_features = np.vstack(features)

    logger.debug("Normalizing vector database")
    faiss.normalize_L2(db_features)

    logger.debug("Building FAISS index")
    features_index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension))

    logger.debug("Adding vector database to FAISS index")
    features_index.add_with_ids(db_features, features_ids)

    logger.debug("Setting number of threads")
    if number_of_threads > faiss.omp_get_max_threads():
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())
    else:
        faiss.omp_set_num_threads(number_of_threads)

    logger.debug("Similarity searching vector database")
    _, results = features_index.search(db_features, k)

    return results


def cluster_records(records: list[vr.VehicleRecord]) -> set[vr.VehicleRecordCluster]:
    # Top K rough search
    vehicle_features = [record.vehicle_feature for record in records]
    vehicle_features_ids = [record.record_id for record in records]
    license_plate_features = [record.license_plate_feature for record in records if
                              record.has_license_plate()]
    license_plate_features_ids = [record.record_id for record in records if record.has_license_plate()]

    vehicle_top_k_results = top_k_search(vehicle_features, vehicle_features_ids, cfg.K, cfg.DIMENSION,
                                         cfg.NUMBER_OF_THREADS)
    license_plate_top_k_results = top_k_search(license_plate_features, license_plate_features_ids, cfg.K,
                                               cfg.DIMENSION,
                                               cfg.NUMBER_OF_THREADS)

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
    for record_id, candidate_records in candidate_records_dict.items():
        record = records_dict[record_id]
        candidate_clusters = {record.cluster for record in candidate_records if record.has_assigned_cluster()}

        if len(candidate_clusters) == 0:
            cluster = vr.VehicleRecordCluster()
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
                cluster = vr.VehicleRecordCluster()
                cluster.add_record(record)

    clusters = {record.cluster for record in records}
    return clusters
