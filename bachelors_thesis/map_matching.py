import logging
import time

import networkx as nx
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

from util import get_path, get_node_path
from vehicle_record import VehicleRecordCluster

logger = logging.getLogger(__name__)


def map_match(clusters: set[VehicleRecordCluster],
              road_graph: nx.MultiDiGraph,
              cameras_info: dict,
              *,
              project=True) -> None:
    road_map = NxMap(parse_osmnx_graph(road_graph, xy=project, network_type=NetworkType.DRIVE))
    skipped_clusters_count = 0
    empty_paths_count = 0
    invalid_paths_count = 0

    t0 = time.time_ns()
    for cluster in clusters:
        if cluster.size() < 2:
            skipped_clusters_count += 1
            continue

        trace = cluster.get_trace(road_graph, cameras_info, project=project)
        matcher = LCSSMatcher(road_map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        if path_df.empty:
            empty_paths_count += 1
            continue

        cluster.path = get_path(road_graph, path_df)
        if cluster.has_valid_path():
            cluster.node_path = get_node_path(cluster.path)
        else:
            invalid_paths_count += 1
    t1 = time.time_ns()

    logger.info(f"Map Matching execution time [ms]: {(t1 - t0) / 1000 / 1000}")
    logger.info(f"Number of skipped clusters: {skipped_clusters_count}")
    logger.info(f"Number of clusters with an empty path: {empty_paths_count}")
    logger.info(f"Number of clusters with an invalid path: {invalid_paths_count}")
    logger.info(
        f"Non-skipped clusters with a valid path: {1 - (skipped_clusters_count + empty_paths_count + invalid_paths_count) / len(clusters)}")
