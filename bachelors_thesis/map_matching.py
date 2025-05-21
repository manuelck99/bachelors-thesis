from __future__ import annotations

import math
from multiprocessing import Pool
from typing import TYPE_CHECKING

import networkx as nx
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

from util import load_graph, get_trace, get_trace_from_list, get_node_path

if TYPE_CHECKING:
    from vehicle_record import Cluster, Record


def map_match_clusters(clusters: set[Cluster],
                       road_graph: nx.MultiDiGraph,
                       cameras_info: dict) -> None:
    clusters_to_map_match = list()
    for cluster in clusters:
        if cluster.get_size() >= 2:
            clusters_to_map_match.append(cluster)

    traces = list()
    for cluster in clusters_to_map_match:
        trace = cluster.get_trace_as_list(road_graph, cameras_info)
        traces.append(trace)

    node_paths = map_match_traces(traces, road_graph)
    for node_path, cluster in zip(node_paths, clusters_to_map_match):
        cluster.set_node_path(node_path)


def map_match_records(records: list[Record], road_graph: nx.MultiDiGraph, cameras_info: dict) -> list[int] | None:
    trace = get_trace(records, road_graph, cameras_info)

    road_map = NxMap(parse_osmnx_graph(road_graph, network_type=NetworkType.DRIVE, xy=True))
    matcher = LCSSMatcher(road_map)
    match_result = matcher.match_trace(trace)
    path_df = match_result.path_to_dataframe()

    if path_df.empty:
        return None
    else:
        return get_node_path(path_df, road_graph)


def map_match_traces(traces: list[list[list[float]]], road_graph: nx.MultiDiGraph) -> list[list[int] | None]:
    number_of_processes = 4
    number_of_traces = len(traces)
    chunk_size = math.ceil(number_of_traces / number_of_processes)
    traces_chunks = [traces[i:i + chunk_size] for i in range(0, number_of_traces, chunk_size)]

    args = [(traces_chunk, road_graph.graph["path"]) for traces_chunk in traces_chunks]
    with Pool(processes=number_of_processes) as pool:
        node_paths = pool.starmap(_map_match_traces, args)
    node_paths = [node_path for node_paths_chunk in node_paths for node_path in node_paths_chunk]

    return node_paths


def _map_match_traces(traces: list[list[list[float]]], road_graph_path: str) -> list[list[int] | None]:
    road_graph = load_graph(road_graph_path)
    road_map = NxMap(parse_osmnx_graph(road_graph, network_type=NetworkType.DRIVE, xy=True))

    node_paths = list()
    for trace in traces:
        trace = get_trace_from_list(trace)
        matcher = LCSSMatcher(road_map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        if path_df.empty:
            node_paths.append(None)
        else:
            node_paths.append(get_node_path(path_df, road_graph))

    return node_paths
