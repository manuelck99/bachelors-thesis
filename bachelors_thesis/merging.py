import math
from multiprocessing import Queue, Process
from uuid import UUID

import networkx as nx

from clustering import TopKSearcher
from config import DIMENSION, K, MERGING_THRESHOLD
from map_matching import map_match_traces
from region import RegionCompact
from vehicle_record import Record, Cluster, VehicleRecordClusterCompact


def find_clusters_to_merge_for_region(aux_region: RegionCompact,
                                      region_i_clusters: set[Cluster],
                                      region_j_clusters: set[Cluster],
                                      region_partitioning: dict,
                                      *,
                                      cameras_info: dict) -> set[tuple[UUID, UUID]]:
    i, j = aux_region.region_id

    aux_clusters_crossing_from_i_to_j = set()
    aux_clusters_crossing_from_j_to_i = set()
    for cluster in aux_region.clusters:
        if cluster_crosses_from_i_to_j(cluster, region_partitioning[aux_region.region_id]["edges_i_to_j"]):
            aux_clusters_crossing_from_i_to_j.add(cluster)

        if cluster_crosses_from_i_to_j(cluster, region_partitioning[aux_region.region_id]["edges_j_to_i"]):
            aux_clusters_crossing_from_j_to_i.add(cluster)

    clusters_to_merge = set()
    aux_region_nodes = region_partitioning[aux_region.region_id]["nodes"]
    overlapping_nodes_i_ij = region_partitioning[i]["nodes"].intersection(aux_region_nodes)
    overlapping_nodes_ij_j = region_partitioning[j]["nodes"].intersection(aux_region_nodes)
    if len(aux_clusters_crossing_from_i_to_j) != 0:
        clusters_to_merge.update(find_clusters_crossing_from_i_to_j_to_merge(aux_clusters_crossing_from_i_to_j,
                                                                             aux_region_nodes,
                                                                             overlapping_nodes_i_ij,
                                                                             overlapping_nodes_ij_j,
                                                                             region_i_clusters,
                                                                             region_j_clusters,
                                                                             cameras_info=cameras_info))

    if len(aux_clusters_crossing_from_j_to_i) != 0:
        clusters_to_merge.update(find_clusters_crossing_from_i_to_j_to_merge(aux_clusters_crossing_from_j_to_i,
                                                                             aux_region_nodes,
                                                                             overlapping_nodes_ij_j,
                                                                             overlapping_nodes_i_ij,
                                                                             region_j_clusters,
                                                                             region_i_clusters,
                                                                             cameras_info=cameras_info))

    return clusters_to_merge


def find_clusters_crossing_from_i_to_j_to_merge(aux_clusters_crossing_from_i_to_j: set[Cluster],
                                                aux_region_nodes: set[int],
                                                overlapping_nodes_i_ij: set[int],
                                                overlapping_nodes_ij_j: set[int],
                                                region_i_clusters: set[Cluster],
                                                region_j_clusters: set[Cluster],
                                                *,
                                                cameras_info: dict) -> set[tuple[UUID, UUID]]:
    aux_clusters_dict = {cluster.get_cluster_id(): cluster for cluster in aux_clusters_crossing_from_i_to_j}
    clusters_to_merge = set()

    clusters_i_ij = set()
    for cluster in region_i_clusters:
        # TODO: Try only ends inside aux region
        # TODO: Try node vs. record
        # TODO: Try at least one node/record in aux region
        if cluster_starts_outside_ends_inside_aux_region(cluster, aux_region_nodes):
            clusters_i_ij.add(cluster)

    if len(clusters_i_ij) != 0:
        results_i_ij = cluster_rough_search(clusters_i_ij,
                                            aux_clusters_crossing_from_i_to_j)

        for cluster_id, candidate_clusters in results_i_ij.items():
            if len(candidate_clusters) == 0:
                continue

            cluster = aux_clusters_dict[cluster_id]

            # TODO: Try filtering out candidate clusters with low similarity first
            top_merging_score = -math.inf
            top_merging_cluster = None
            for candidate_cluster in candidate_clusters:
                merging_score = calculate_cluster_merging_score(candidate_cluster,
                                                                cluster,
                                                                overlapping_nodes_i_ij,
                                                                cameras_info=cameras_info)
                if merging_score > top_merging_score:
                    top_merging_score = merging_score
                    top_merging_cluster = candidate_cluster

            if top_merging_score >= MERGING_THRESHOLD:
                clusters_to_merge.add((cluster_id, top_merging_cluster.get_cluster_id()))

    clusters_ij_j = set()
    for cluster in region_j_clusters:
        # TODO: Try only starts inside aux region
        # TODO: Try node vs. record
        # TODO: Try at least one node/record in aux region
        if cluster_starts_inside_ends_outside_aux_region(cluster, aux_region_nodes):
            clusters_ij_j.add(cluster)

    if len(clusters_ij_j) != 0:
        results_ij_j = cluster_rough_search(clusters_ij_j,
                                            aux_clusters_crossing_from_i_to_j)

        for cluster_id, candidate_clusters in results_ij_j.items():
            if len(candidate_clusters) == 0:
                continue

            cluster = aux_clusters_dict[cluster_id]

            # TODO: Try filtering out candidate clusters with low similarity first
            top_merging_score = -math.inf
            top_merging_cluster = None
            for candidate_cluster in candidate_clusters:
                merging_score = calculate_cluster_merging_score(cluster,
                                                                candidate_cluster,
                                                                overlapping_nodes_ij_j,
                                                                cameras_info=cameras_info)
                if merging_score > top_merging_score:
                    top_merging_score = merging_score
                    top_merging_cluster = candidate_cluster

            if top_merging_score >= MERGING_THRESHOLD:
                clusters_to_merge.add((cluster_id, top_merging_cluster.get_cluster_id()))

    return clusters_to_merge


def cluster_rough_search(xb_clusters: set[Cluster],
                         xq_clusters: set[Cluster]) -> dict[UUID, set[Cluster]]:
    xb_clusters = list(xb_clusters)
    xq_clusters = list(xq_clusters)

    vehicle_features_xb = [cluster.get_centroid_vehicle_feature() for cluster in xb_clusters]
    vehicle_features_ids_xb = [cluster.get_cluster_id() for cluster in xb_clusters]
    license_plate_features_xb = [cluster.get_centroid_license_plate_feature() for cluster in
                                 filter(lambda c: c.has_license_plate(), xb_clusters)]
    license_plate_features_ids_xb = [cluster.get_cluster_id() for cluster in
                                     filter(lambda c: c.has_license_plate(), xb_clusters)]

    vehicle_features_xq = [cluster.get_centroid_vehicle_feature() for cluster in xq_clusters]
    vehicle_features_ids_xq = [cluster.get_cluster_id() for cluster in xq_clusters]
    license_plate_features_xq = [cluster.get_centroid_license_plate_feature() for cluster in
                                 filter(lambda c: c.has_license_plate(), xq_clusters)]
    license_plate_features_ids_xq = [cluster.get_cluster_id() for cluster in
                                     filter(lambda c: c.has_license_plate(), xq_clusters)]

    vehicle_top_k_searcher = TopKSearcher(vehicle_features_xb,
                                          vehicle_features_ids_xb,
                                          dimension=DIMENSION)
    vehicle_top_k_results = vehicle_top_k_searcher.search_on_all_gpus(vehicle_features_xq,
                                                                      vehicle_features_ids_xq,
                                                                      k=K)

    license_plate_top_k_searcher = TopKSearcher(license_plate_features_xb,
                                                license_plate_features_ids_xb,
                                                dimension=DIMENSION)
    license_plate_top_k_results = license_plate_top_k_searcher.search_on_all_gpus(license_plate_features_xq,
                                                                                  license_plate_features_ids_xq,
                                                                                  k=K)

    xb_clusters_dict = {cluster.get_cluster_id(): cluster for cluster in xb_clusters}
    candidate_clusters_dict = dict()
    for cluster_id, top_k_ids in vehicle_top_k_results.items():
        s = {xb_clusters_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != cluster_id}
        candidate_clusters_dict[cluster_id] = s

    for cluster_id, top_k_ids in license_plate_top_k_results.items():
        s = {xb_clusters_dict[top_k_id] for top_k_id in top_k_ids if top_k_id != cluster_id}
        candidate_clusters_dict[cluster_id] = candidate_clusters_dict[cluster_id].union(s)

    return candidate_clusters_dict


def cluster_crosses_from_i_to_j(cluster: Cluster,
                                edges_from_i_to_j: set[tuple[int, int]]) -> bool:
    for u, v in zip(cluster.get_node_path(), cluster.get_node_path()[1:]):
        if (u, v) in edges_from_i_to_j:
            return True
    return False


def cluster_starts_outside_ends_inside_aux_region(cluster: Cluster,
                                                  aux_region_nodes: set[int]) -> bool:
    node_path = cluster.get_node_path()
    start_node = node_path[0]
    end_node = node_path[-1]
    return start_node not in aux_region_nodes and end_node in aux_region_nodes


def cluster_starts_inside_ends_outside_aux_region(cluster: Cluster,
                                                  aux_region_nodes: set[int]) -> bool:
    node_path = cluster.get_node_path()
    start_node = node_path[0]
    end_node = node_path[-1]
    return start_node in aux_region_nodes and end_node not in aux_region_nodes


def calculate_cluster_merging_score(cluster_from: Cluster,
                                    cluster_to: Cluster,
                                    overlapping_nodes: set[int],
                                    *,
                                    gamma=0.8,
                                    cameras_info: dict) -> float:
    cluster_from_records = {record for record in
                            filter(lambda r: is_record_in_nodes(r, overlapping_nodes, cameras_info),
                                   cluster_from.get_records())}
    cluster_to_records = {record for record in
                          filter(lambda r: is_record_in_nodes(r, overlapping_nodes, cameras_info),
                                 cluster_to.get_records())}

    a = len(cluster_from_records.intersection(cluster_to_records))
    b = len(cluster_from_records.union(cluster_to_records)) - a

    if a > 0:
        return math.pow(gamma, b) * math.exp(a) / (1 + math.exp(a))
    else:
        # TODO: Try Dijkstra (shortest path based on length or duration) or euclidean and/or time distance
        # TODO: Try adding historical data for probability- or ML-model
        return -math.inf


def is_record_in_nodes(record: Record, nodes: set[int], cameras_info: dict) -> bool:
    camera_id = record.get_camera_id()
    camera = cameras_info[camera_id]
    node_id = camera["node_id"]
    return node_id in nodes


def merge_clusters(clusters: dict[UUID, Cluster],
                   clusters_to_merge: set[tuple[UUID, UUID]],
                   *,
                   road_graph: nx.MultiDiGraph,
                   cameras_info: dict,
                   number_of_processes=4) -> dict[UUID, Cluster]:
    merging_graph = nx.Graph()
    for i, j in clusters_to_merge:
        merging_graph.add_edge(i, j)

    merged_clusters = list()
    for clusters_ids_component in nx.connected_components(merging_graph):
        clusters_component = {clusters[cluster_id] for cluster_id in clusters_ids_component}

        for cluster_id in clusters_ids_component:
            del clusters[cluster_id]

        merged_clusters.append(clusters_component)

    chunk_size = math.ceil(len(merged_clusters) / number_of_processes)
    merged_clusters_chunks = [merged_clusters[i:i + chunk_size] for i in range(0, len(merged_clusters), chunk_size)]
    results_queue = Queue()
    args = [(merged_clusters_chunk, road_graph, cameras_info, results_queue) for merged_clusters_chunk in
            merged_clusters_chunks]
    processes = list()
    for arg in args:
        process = Process(target=_merge_clusters, args=arg)
        processes.append(process)

    for process in processes:
        process.start()

    merged_clusters = list()
    for _ in range(number_of_processes):
        merged_clusters_chunk = results_queue.get()
        merged_clusters.extend(merged_clusters_chunk)

    for process in processes:
        process.join()

    for merged_cluster in merged_clusters:
        clusters[merged_cluster.get_cluster_id()] = merged_cluster

    return clusters


def _merge_clusters(clusters_to_merge: list[set[Cluster]],
                    road_graph: nx.MultiDiGraph,
                    cameras_info: dict,
                    results_queue: Queue) -> None:
    clusters = list()
    for clusters_to_merge_temp in clusters_to_merge:
        cluster = VehicleRecordClusterCompact.from_clusters_partial(clusters_to_merge_temp)
        clusters.append(cluster)

    traces = list()
    for cluster in clusters:
        trace = cluster.get_trace_as_list(road_graph, cameras_info)
        traces.append(trace)

    node_paths = map_match_traces(traces, road_graph)
    for node_path, cluster in zip(node_paths, clusters):
        cluster.set_node_path(node_path)

    results_queue.put(clusters)
