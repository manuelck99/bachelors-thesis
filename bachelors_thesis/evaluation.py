import math
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

from util import get_trace, get_path, get_node_path, EPSG_32650
from vehicle_record import VehicleRecord, VehicleRecordCluster

Precision = float
Recall = float
F1_Score = float
Expansion = float

LCSS_Distance = float
EDR_Distance = float
STLC_Distance = float


# Spatio-Temporal Vehicle Trajectory Recovery on Road Network Based on Traffic Camera Video Data
def yu_ao_yan_cluster_evaluation(records: list[VehicleRecord],
                                 clusters: set[VehicleRecordCluster]) -> tuple[Precision, Recall, F1_Score, Expansion]:
    vehicle_records_count = defaultdict(int)
    for record in records:
        if record.is_annotated():
            vehicle_records_count[record.vehicle_id] += 1

    precision = 0.0
    recall = 0.0
    for vehicle_id in vehicle_records_count.keys():
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                    cluster_of_vehicle)
        precision += number_of_records_of_vehicle_in_cluster / len(cluster_of_vehicle.records)
        recall += number_of_records_of_vehicle_in_cluster / vehicle_records_count[vehicle_id]

    number_of_annotated_vehicles = len(vehicle_records_count)
    precision /= number_of_annotated_vehicles
    recall /= number_of_annotated_vehicles
    f1_score = (precision * recall) / (precision + recall)

    expansion = 0.0
    for vehicle_id in vehicle_records_count.keys():
        for cluster in clusters:
            number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                        cluster)
            if number_of_records_of_vehicle_in_cluster != 0:
                expansion += 1.0
    expansion /= number_of_annotated_vehicles

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
        if record.is_annotated() and record.vehicle_id == vehicle_id:
            count += 1
    return count


# A survey of trajectory distance measures and performance evaluation
def su_liu_zheng_trajectory_evaluation(records: list[VehicleRecord],
                                       clusters: set[VehicleRecordCluster],
                                       road_graph: nx.MultiDiGraph,
                                       cameras_info: dict,
                                       *,
                                       project=True) -> tuple[LCSS_Distance, EDR_Distance, STLC_Distance]:
    vehicle_records_dict = defaultdict(list)
    for record in records:
        if record.is_annotated():
            vehicle_records_dict[record.vehicle_id].append(record)

    for vehicle_records in vehicle_records_dict.values():
        vehicle_records.sort(key=lambda r: r.timestamp)

    traces_dict = dict()
    for vehicle_id, vehicle_records in vehicle_records_dict.items():
        trace = get_trace(vehicle_records, road_graph, cameras_info, project=project)
        traces_dict[vehicle_id] = trace

    road_map = NxMap(parse_osmnx_graph(road_graph, NetworkType.DRIVE, xy=project))
    node_paths_dict = dict()
    for vehicle_id, trace in traces_dict.items():
        matcher = LCSSMatcher(road_map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        if path_df.empty:
            continue

        path = get_path(road_graph, path_df)
        if path is not None and len(path) > 0:
            node_paths_dict[vehicle_id] = get_node_path(path)

    road_graph = ox.project_graph(road_graph, to_crs=EPSG_32650)
    lcss = 0.0
    edr = 0.0
    for vehicle_id, node_path in node_paths_dict.items():
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)

        lcss += lcss_distance(cluster_of_vehicle.node_path, node_path, road_graph, epsilon=10)
        edr += edr_distance(cluster_of_vehicle.node_path, node_path, road_graph, epsilon=10)

    number_of_paths = len(node_paths_dict)
    lcss /= number_of_paths
    edr /= number_of_paths

    return lcss, edr, 0.0


def lcss_distance(
        trajectory: list[int],
        trajectory_gt: list[int],
        road_graph: nx.MultiDiGraph,
        *,
        epsilon: float) -> float:
    trajectory = np.array(trajectory)
    trajectory_gt = np.array(trajectory_gt)

    n, m = len(trajectory), len(trajectory_gt)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n):
        for j in range(m):
            node_id = trajectory[i]
            node_id_gt = trajectory_gt[j]

            x, y = road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]
            x_gt, y_gt = road_graph.nodes[node_id_gt]["x"], road_graph.nodes[node_id_gt]["y"]

            distance = ox.distance.euclidean(y, x, y_gt, x_gt)
            if distance <= epsilon:
                dp[i + 1, j + 1] = dp[i, j] + 1
            else:
                dp[i + 1, j + 1] = max(dp[i, j + 1], dp[i + 1, j])

    lcss = dp[n, m]
    # return lcss
    # return 1 - lcss / (n + m - lcss)
    # return 1 - lcss / m
    return 1 - lcss / min(n, m)


def edr_distance(
        trajectory: list[int],
        trajectory_gt: list[int],
        road_graph: nx.MultiDiGraph,
        *,
        epsilon: float) -> float:
    trajectory = np.array(trajectory)
    trajectory_gt = np.array(trajectory_gt)

    n, m = len(trajectory), len(trajectory_gt)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            node_id = trajectory[i - 1]
            node_id_gt = trajectory_gt[j - 1]

            x, y = road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]
            x_gt, y_gt = road_graph.nodes[node_id_gt]["x"], road_graph.nodes[node_id_gt]["y"]

            distance = ox.distance.euclidean(y, x, y_gt, x_gt)
            if distance <= epsilon:
                subcost = 0
            else:
                subcost = 1
            dp[i, j] = min(dp[i - 1, j - 1] + subcost, dp[i - 1, j] + 1, dp[i, j - 1] + 1)

    edr = dp[n, m]
    # return edr
    # return edr / m
    return edr / max(n, m)


def stlc_distance() -> STLC_Distance:
    pass
