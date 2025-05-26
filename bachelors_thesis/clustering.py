import math
import time
from uuid import UUID

import numpy as np
from faiss import IndexFlatIP, normalize_L2, omp_get_max_threads, omp_set_num_threads, \
    index_cpu_to_all_gpus, index_cpu_to_gpu, StandardGpuResources

from config import K, DIMENSION, NUMBER_OF_THREADS, SIMILARITY_THRESHOLD, WEIGHT_VEHICLE_SIMILARITY, \
    WEIGHT_LICENSE_PLATE_SIMILARITY
from util import log_info
from vehicle_record import VehicleRecord, VehicleRecordCluster


# TODO: Try using a less exact, but faster FAISS index
class TopKSearcher:
    dimension: int
    xb: np.ndarray
    xb_id_map: dict[int, UUID]

    def __init__(self,
                 features: list[np.ndarray],
                 features_ids: list[UUID],
                 *,
                 dimension: int):
        self.dimension = dimension
        self.xb = np.vstack(features)
        normalize_L2(self.xb)
        self.xb_id_map = {i: feature_id for i, feature_id in enumerate(features_ids)}

    def search(self,
               features: list[np.ndarray],
               features_ids: list[UUID],
               *,
               k: int,
               number_of_threads: int) -> dict[UUID, list[UUID]]:
        xq = np.vstack(features)
        normalize_L2(xq)

        if number_of_threads > omp_get_max_threads():
            omp_set_num_threads(omp_get_max_threads())
        else:
            omp_set_num_threads(number_of_threads)

        index = IndexFlatIP(self.dimension)
        index.add(self.xb)
        _, results = index.search(xq, k)

        xq_id_map = dict()
        for feature_id, top_k_ids in zip(features_ids, results):
            xq_id_map[feature_id] = [self.xb_id_map[top_k_id] for top_k_id in top_k_ids if top_k_id != -1]

        return xq_id_map

    def search_on_all_gpus(self,
                           features: list[np.ndarray],
                           features_ids: list[UUID],
                           *,
                           k: int) -> dict[UUID, list[UUID]]:
        xq = np.vstack(features)
        normalize_L2(xq)

        index = IndexFlatIP(self.dimension)
        index = index_cpu_to_all_gpus(index)
        index.add(self.xb)
        _, results = index.search(xq, k)

        xq_id_map = dict()
        for feature_id, top_k_ids in zip(features_ids, results):
            xq_id_map[feature_id] = [self.xb_id_map[top_k_id] for top_k_id in top_k_ids if top_k_id != -1]

        return xq_id_map

    def search_on_gpu(self,
                      features: list[np.ndarray],
                      features_ids: list[UUID],
                      *,
                      k: int,
                      gpu: int) -> dict[UUID, list[UUID]]:
        xq = np.vstack(features)
        normalize_L2(xq)

        index = IndexFlatIP(self.dimension)
        index = index_cpu_to_gpu(StandardGpuResources(), gpu, index)
        index.add(self.xb)
        _, results = index.search(xq, k)

        xq_id_map = dict()
        for feature_id, top_k_ids in zip(features_ids, results):
            xq_id_map[feature_id] = [self.xb_id_map[top_k_id] for top_k_id in top_k_ids if top_k_id != -1]

        return xq_id_map


def cluster_records(records: list[VehicleRecord],
                    *,
                    region=None,
                    lock=None,
                    use_gpu=False) -> set[VehicleRecordCluster]:
    # Top K rough search
    t0 = time.time_ns()
    vehicle_features = [record.get_vehicle_feature() for record in records]
    vehicle_features_ids = [record.get_record_id() for record in records]
    license_plate_features = [record.get_license_plate_feature() for record in
                              filter(lambda r: r.has_license_plate(), records)]
    license_plate_features_ids = [record.get_record_id() for record in
                                  filter(lambda r: r.has_license_plate(), records)]

    vehicle_top_k_searcher = TopKSearcher(vehicle_features, vehicle_features_ids, dimension=DIMENSION)
    if use_gpu:
        vehicle_top_k_results = vehicle_top_k_searcher.search_on_all_gpus(vehicle_features,
                                                                          vehicle_features_ids,
                                                                          k=K)
    else:
        vehicle_top_k_results = vehicle_top_k_searcher.search(vehicle_features,
                                                              vehicle_features_ids,
                                                              k=K,
                                                              number_of_threads=NUMBER_OF_THREADS)

    license_plate_top_k_searcher = TopKSearcher(license_plate_features, license_plate_features_ids, dimension=DIMENSION)
    if use_gpu:
        license_plate_top_k_results = license_plate_top_k_searcher.search_on_all_gpus(license_plate_features,
                                                                                      license_plate_features_ids,
                                                                                      k=K)
    else:
        license_plate_top_k_results = license_plate_top_k_searcher.search(license_plate_features,
                                                                          license_plate_features_ids,
                                                                          k=K,
                                                                          number_of_threads=NUMBER_OF_THREADS)
    t1 = time.time_ns()
    log_info(f"Top K rough search time [ms]: {(t1 - t0) / 1000 / 1000}", region=region, lock=lock)

    # Merging rough search results for vehicle features and license plate features
    t0 = time.time_ns()
    records_dict = {record.get_record_id(): record for record in records}
    candidate_records_dict = dict()
    for record_id, top_k_ids in vehicle_top_k_results.items():
        s = {records_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != record_id}
        candidate_records_dict[record_id] = s

    for record_id, top_k_ids in license_plate_top_k_results.items():
        s = {records_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != record_id}
        candidate_records_dict[record_id] = candidate_records_dict[record_id].union(s)
    t1 = time.time_ns()
    log_info(f"Merging rough search results time [ms]: {(t1 - t0) / 1000 / 1000}", region=region, lock=lock)

    # Clustering
    # TODO: Try using spatio-temporal data
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
    log_info(f"Clustering time [ms]: {(t1 - t0) / 1000 / 1000}", region=region, lock=lock)

    clusters = {record.get_cluster() for record in records if record.has_assigned_cluster()}
    singleton_clusters = {cluster for cluster in clusters if cluster.get_size() == 1}
    log_info(f"Number of clusters: {len(clusters)}", region=region, lock=lock)
    log_info(f"Number of singleton clusters: {len(singleton_clusters)}", region=region, lock=lock)
    log_info(f"Number of non-singleton clusters: {len(clusters) - len(singleton_clusters)}", region=region, lock=lock)

    return clusters
