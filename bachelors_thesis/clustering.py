import logging
import math
import time
from uuid import UUID

import numpy as np
from faiss import IndexFlatIP, normalize_L2, index_cpu_to_all_gpus, omp_get_max_threads, omp_set_num_threads

from config import K, DIMENSION, NUMBER_OF_THREADS, SIMILARITY_THRESHOLD, WEIGHT_VEHICLE_SIMILARITY, \
    WEIGHT_LICENSE_PLATE_SIMILARITY
from vehicle_record import VehicleRecord, VehicleRecordCluster

logger = logging.getLogger(__name__)


class TopKSearcher:
    xb: np.ndarray
    index: IndexFlatIP
    xb_id_map: dict[int, UUID]

    def __init__(self,
                 features: list[np.ndarray],
                 features_ids: list[UUID],
                 *,
                 dimension: int):
        self.xb = np.vstack(features)
        normalize_L2(self.xb)
        self.dimension = dimension
        self.index = IndexFlatIP(self.dimension)
        self.xb_id_map = {i: feature_id for i, feature_id in enumerate(features_ids)}

    def search(self,
               features: list[np.ndarray],
               features_ids: list[UUID],
               *,
               k: int,
               number_of_threads: int,
               use_gpu: bool) -> dict[UUID, list[UUID]]:
        xq = np.vstack(features)
        normalize_L2(xq)

        if use_gpu:
            self.index = index_cpu_to_all_gpus(self.index)
        else:
            if number_of_threads > omp_get_max_threads():
                omp_set_num_threads(omp_get_max_threads())
            else:
                omp_set_num_threads(number_of_threads)

        self.index.add(self.xb)
        _, results = self.index.search(xq, k)

        xq_id_map = dict()
        for feature_id, top_k_ids in zip(features_ids, results):
            xq_id_map[feature_id] = [self.xb_id_map[top_k_id] for top_k_id in top_k_ids if top_k_id != -1]

        return xq_id_map


def cluster_records(records: list[VehicleRecord], *, use_gpu=False) -> set[VehicleRecordCluster]:
    # Top K rough search
    vehicle_features = [record.get_vehicle_feature() for record in records]
    vehicle_features_ids = [record.get_record_id() for record in records]
    license_plate_features = [record.get_license_plate_feature() for record in
                              filter(lambda r: r.has_license_plate(), records)]
    license_plate_features_ids = [record.get_record_id() for record in
                                  filter(lambda r: r.has_license_plate(), records)]

    t0 = time.time_ns()
    vehicle_top_k_searcher = TopKSearcher(vehicle_features, vehicle_features_ids, dimension=DIMENSION)
    vehicle_top_k_results = vehicle_top_k_searcher.search(vehicle_features,
                                                          vehicle_features_ids,
                                                          k=K,
                                                          number_of_threads=NUMBER_OF_THREADS,
                                                          use_gpu=use_gpu)

    license_plate_top_k_searcher = TopKSearcher(license_plate_features, license_plate_features_ids, dimension=DIMENSION)
    license_plate_top_k_results = license_plate_top_k_searcher.search(license_plate_features,
                                                                      license_plate_features_ids,
                                                                      k=K,
                                                                      number_of_threads=NUMBER_OF_THREADS,
                                                                      use_gpu=use_gpu)
    t1 = time.time_ns()
    logger.info(f"Top K search time [ms]: {(t1 - t0) / 1000 / 1000}")

    # Merging rough search results for vehicle features and license plate features
    records_dict = {record.get_record_id(): record for record in records}
    candidate_records_dict = dict()
    for record_id, top_k_ids in vehicle_top_k_results.items():
        s = {records_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != record_id}
        candidate_records_dict[record_id] = s

    for record_id, top_k_ids in license_plate_top_k_results.items():
        s = {records_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != record_id}
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
