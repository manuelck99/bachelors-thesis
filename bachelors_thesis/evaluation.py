import json
import math
from collections import defaultdict

import networkx as nx
import numpy as np
import osmnx as ox

from config import NUMBER_OF_ANNOTATED_VEHICLES
from map_matching import map_match_traces
from util import EPSG_32650, get_trace_as_list
from vehicle_record import Record, Cluster, VehicleRecordCluster, VehicleRecordClusterCompact

Precision = float
Recall = float
F1_Score = float
Expansion = float

LCSS_Distance = float
EDR_Distance = float
STLC_Distance = float


def cluster_evaluation_with_record_gt(records_gt: list[Record],
                                      clusters: set[Cluster]) -> tuple[Precision, Recall, F1_Score, Expansion]:
    vehicle_records_count = defaultdict(int)
    for record_gt in records_gt:
        if record_gt.is_annotated():
            vehicle_records_count[record_gt.get_vehicle_id()] += 1

    precision = 0.0
    recall = 0.0
    expansion = 0.0
    skipped_vehicles_count = 0
    for vehicle_id in vehicle_records_count.keys():
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        if cluster_of_vehicle is None:
            skipped_vehicles_count += 1
            continue

        number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                    cluster_of_vehicle)
        precision += number_of_records_of_vehicle_in_cluster / cluster_of_vehicle.get_size()
        recall += number_of_records_of_vehicle_in_cluster / vehicle_records_count[vehicle_id]

        for cluster in clusters:
            number_of_records_of_vehicle_in_cluster = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id,
                                                                                                        cluster)
            if number_of_records_of_vehicle_in_cluster != 0:
                expansion += 1.0

    number_of_annotated_vehicles = len(vehicle_records_count) - skipped_vehicles_count
    precision /= number_of_annotated_vehicles
    recall /= number_of_annotated_vehicles
    f1_score = (precision * recall) / (precision + recall)
    expansion /= number_of_annotated_vehicles

    return precision, recall, f1_score, expansion


def cluster_evaluation_with_cluster_gt(clusters_gt: set[Cluster],
                                       clusters: set[Cluster]) -> tuple[Precision, Recall, F1_Score, Expansion]:
    precision = 0.0
    recall = 0.0
    expansion = 0.0
    skipped_vehicles_count = 0
    for vehicle_id in range(NUMBER_OF_ANNOTATED_VEHICLES):
        cluster_of_vehicle_gt = find_cluster_of_vehicle(vehicle_id, clusters_gt)
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        if cluster_of_vehicle_gt is None or cluster_of_vehicle is None:
            skipped_vehicles_count += 1
            continue

        records_in_common = cluster_of_vehicle_gt.get_records().intersection(cluster_of_vehicle.get_records())

        precision += len(records_in_common) / cluster_of_vehicle.get_size()
        recall += len(records_in_common) / cluster_of_vehicle_gt.get_size()

        for cluster in clusters:
            if len(cluster_of_vehicle_gt.get_records().intersection(cluster.get_records())) != 0:
                expansion += 1.0

    number_of_annotated_vehicles = NUMBER_OF_ANNOTATED_VEHICLES - skipped_vehicles_count
    precision /= number_of_annotated_vehicles
    recall /= number_of_annotated_vehicles
    f1_score = (precision * recall) / (precision + recall)
    expansion /= number_of_annotated_vehicles

    return precision, recall, f1_score, expansion


def trajectory_evaluation_with_record_gt(records_gt: list[Record],
                                         clusters: set[Cluster],
                                         road_graph: nx.MultiDiGraph,
                                         road_graph_path: str,
                                         cameras_info: dict,
                                         *,
                                         gamma: float,
                                         epsilon: float) -> tuple[LCSS_Distance, EDR_Distance, STLC_Distance]:
    vehicle_records_dict = defaultdict(list)
    for record_gt in records_gt:
        if record_gt.is_annotated():
            vehicle_records_dict[record_gt.get_vehicle_id()].append(record_gt)

    for vehicle_records in vehicle_records_dict.values():
        vehicle_records.sort(key=lambda r: r.get_timestamp())

    road_graph_proj = ox.project_graph(road_graph, to_crs=EPSG_32650)
    stlc = 0.0
    skipped_vehicles_count = 0
    for vehicle_id in vehicle_records_dict.keys():
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        if cluster_of_vehicle is None:
            skipped_vehicles_count += 1
            continue

        trajectory = cluster_of_vehicle.get_ordered_records()
        trajectory_gt = vehicle_records_dict[vehicle_id]
        stlc += stlc_distance(trajectory, trajectory_gt, road_graph_proj, cameras_info, gamma=gamma)
    stlc /= len(vehicle_records_dict) - skipped_vehicles_count

    traces_dict = dict()
    for vehicle_id, vehicle_records in vehicle_records_dict.items():
        trace = get_trace_as_list(vehicle_records, road_graph, cameras_info)
        traces_dict[vehicle_id] = trace

    vehicle_ids = list()
    traces = list()
    for vehicle_id, trace in traces_dict.items():
        vehicle_ids.append(vehicle_id)
        traces.append(trace)

    node_paths_dict = dict()
    node_paths = map_match_traces(traces, road_graph_path)
    for vehicle_id, node_path in zip(vehicle_ids, node_paths):
        if node_path is not None:
            node_paths_dict[vehicle_id] = node_path

    lcss = 0.0
    edr = 0.0
    skipped_vehicles_count = 0
    for vehicle_id, node_path in node_paths_dict.items():
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)

        if cluster_of_vehicle is not None and cluster_of_vehicle.has_valid_node_path():
            lcss += lcss_distance(cluster_of_vehicle.get_node_path(), node_path, road_graph_proj, epsilon=epsilon)
            edr += edr_distance(cluster_of_vehicle.get_node_path(), node_path, road_graph_proj, epsilon=epsilon)
        else:
            skipped_vehicles_count += 1

    number_of_paths = len(node_paths_dict) - skipped_vehicles_count
    lcss /= number_of_paths
    edr /= number_of_paths

    return lcss, edr, stlc


def trajectory_evaluation_with_cluster_gt(clusters_gt: set[Cluster],
                                          clusters: set[Cluster],
                                          road_graph: nx.MultiDiGraph,
                                          cameras_info: dict,
                                          *,
                                          gamma: float,
                                          epsilon: float) -> tuple[
    LCSS_Distance, EDR_Distance, STLC_Distance]:
    road_graph_proj = ox.project_graph(road_graph, to_crs=EPSG_32650)
    stlc = 0.0
    skipped_vehicles_count = 0
    for vehicle_id in range(NUMBER_OF_ANNOTATED_VEHICLES):
        cluster_of_vehicle_gt = find_cluster_of_vehicle(vehicle_id, clusters_gt)
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)
        if cluster_of_vehicle_gt is None or cluster_of_vehicle is None:
            skipped_vehicles_count += 1
            continue

        trajectory_gt = cluster_of_vehicle_gt.get_ordered_records()
        trajectory = cluster_of_vehicle.get_ordered_records()
        stlc += stlc_distance(trajectory, trajectory_gt, road_graph_proj, cameras_info, gamma=gamma)
    stlc /= NUMBER_OF_ANNOTATED_VEHICLES - skipped_vehicles_count

    lcss = 0.0
    edr = 0.0
    skipped_vehicles_count = 0
    for vehicle_id in range(NUMBER_OF_ANNOTATED_VEHICLES):
        cluster_of_vehicle_gt = find_cluster_of_vehicle(vehicle_id, clusters_gt)
        cluster_of_vehicle = find_cluster_of_vehicle(vehicle_id, clusters)

        if (cluster_of_vehicle_gt is not None and cluster_of_vehicle_gt.has_valid_node_path()
                and cluster_of_vehicle is not None and cluster_of_vehicle.has_valid_node_path()):
            lcss += lcss_distance(cluster_of_vehicle.get_node_path(),
                                  cluster_of_vehicle_gt.get_node_path(),
                                  road_graph_proj,
                                  epsilon=epsilon)
            edr += edr_distance(cluster_of_vehicle.get_node_path(),
                                cluster_of_vehicle_gt.get_node_path(),
                                road_graph_proj,
                                epsilon=epsilon)
        else:
            skipped_vehicles_count += 1

    number_of_paths = NUMBER_OF_ANNOTATED_VEHICLES - skipped_vehicles_count
    lcss /= number_of_paths
    edr /= number_of_paths

    return lcss, edr, stlc


def find_cluster_of_vehicle(vehicle_id: int, clusters: set[Cluster]) -> Cluster | None:
    max_number_of_records = -math.inf
    max_number_of_records_cluster = None
    for cluster in clusters:
        number_of_records = calculate_number_of_records_of_vehicle_in_cluster(vehicle_id, cluster)
        if number_of_records > max_number_of_records:
            max_number_of_records = number_of_records
            max_number_of_records_cluster = cluster
    return max_number_of_records_cluster


def calculate_number_of_records_of_vehicle_in_cluster(vehicle_id: int, cluster: Cluster) -> int:
    count = 0
    for record in cluster.get_records():
        if record.is_annotated() and record.get_vehicle_id() == vehicle_id:
            count += 1
    return count


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
    assert lcss <= min(n, m)
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
    assert edr <= max(n, m)
    return edr / max(n, m)


def stlc_distance(trajectory: list[Record],
                  trajectory_gt: list[Record],
                  road_graph: nx.MultiDiGraph,
                  cameras_info: dict,
                  *,
                  gamma: float) -> float:
    spatial_similarity = calculate_spatio_temporal_similarity(trajectory, trajectory_gt, road_graph, cameras_info,
                                                              spatial=True) + calculate_spatio_temporal_similarity(
        trajectory_gt, trajectory, road_graph, cameras_info, spatial=True)

    temporal_similarity = calculate_spatio_temporal_similarity(trajectory, trajectory_gt, road_graph, cameras_info,
                                                               spatial=False) + calculate_spatio_temporal_similarity(
        trajectory_gt, trajectory, road_graph, cameras_info, spatial=False)

    stlc = gamma * spatial_similarity + (1 - gamma) * temporal_similarity
    assert stlc <= 2.0
    return 1 - stlc / 2.0


def calculate_spatio_temporal_similarity(trajectory_1: list[Record],
                                         trajectory_2: list[Record],
                                         road_graph: nx.MultiDiGraph,
                                         cameras_info: dict,
                                         *,
                                         spatial: bool) -> float:
    if spatial:
        f = lambda p, t: calculate_spatial_distance(p, t, road_graph, cameras_info)
    else:
        f = calculate_temporal_distance

    similarity = 0.0
    for point in trajectory_1:
        similarity += math.exp(-f(point, trajectory_2))
    return similarity / len(trajectory_1)


def calculate_spatial_distance(point: Record,
                               trajectory: list[Record],
                               road_graph: nx.MultiDiGraph,
                               cameras_info: dict) -> float:
    min_distance = math.inf
    x, y = point.get_coordinates(road_graph, cameras_info)
    for point_ in trajectory:
        x_, y_ = point_.get_coordinates(road_graph, cameras_info)
        distance = ox.distance.euclidean(y, x, y_, x_)
        if distance < min_distance:
            min_distance = distance
    return min_distance


def calculate_temporal_distance(point: Record, trajectory: list[Record]) -> float:
    min_distance = math.inf
    timestamp = point.get_timestamp()
    for point_ in trajectory:
        distance = abs(timestamp - point_.get_timestamp())
        if distance < min_distance:
            min_distance = distance
    return min_distance


def save_vehicle_clusters(clusters: set[VehicleRecordCluster], path: str) -> None:
    vehicle_clusters = list()
    for vehicle_id in range(NUMBER_OF_ANNOTATED_VEHICLES):
        cluster_of_vehicle: VehicleRecordCluster | None = find_cluster_of_vehicle(vehicle_id, clusters)
        if cluster_of_vehicle is not None:
            vehicle_clusters.append(cluster_of_vehicle.to_dict())

    with open(path, mode="w", encoding="utf-8") as file:
        json.dump(vehicle_clusters, file)


def load_vehicle_clusters(path: str) -> set[VehicleRecordClusterCompact]:
    vehicle_clusters = None
    with open(path, mode="r", encoding="utf-8") as file:
        vehicle_clusters = json.load(file)

    return {VehicleRecordClusterCompact.from_dict(vehicle_cluster) for vehicle_cluster in vehicle_clusters}
