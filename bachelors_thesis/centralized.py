import logging
import random
import time
from argparse import ArgumentParser

from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

from clustering import cluster_records
from evaluation import yu_ao_yan_evaluation
from util import load, load_graph, get_path
from vehicle_record import load_records

logger = logging.getLogger(__name__)


def run(records_path: str, road_graph_path: str, cameras_info_path: str, use_gpu: bool) -> None:
    records = load_records(records_path)
    logger.info(f"Number of records: {len(records)}")

    random.shuffle(records)

    clusters = cluster_records(records, use_gpu=use_gpu)
    logger.info(f"Number of clusters: {len(clusters)}")

    precision, recall, f1_score, expansion = yu_ao_yan_evaluation(records, clusters)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")

    t0 = time.time_ns()
    road_graph = load_graph(road_graph_path)
    road_map = NxMap(parse_osmnx_graph(road_graph, xy=True, network_type=NetworkType.DRIVE))
    cameras_info: dict = load(cameras_info_path)
    empty_paths_count = 0
    invalid_paths_count = 0
    for cluster in clusters:
        if cluster.size() < 3:
            continue

        trace = cluster.get_trace(road_graph, cameras_info)
        matcher = LCSSMatcher(road_map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        if path_df.empty:
            empty_paths_count += 1
            continue

        path = get_path(road_graph, path_df)
        if path is not None and len(path) > 0:
            cluster.path = path
        else:
            invalid_paths_count += 1
    t1 = time.time_ns()
    logger.info(f"Map Matching execution time [ms]: {(t1 - t0) / 1000 / 1000}")
    logger.info(f"Number of clusters with an empty path: {empty_paths_count}")
    logger.info(f"Number of clusters with an invalid path: {invalid_paths_count}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input-path",
        type=str,
        required=True,
        help="Path to the input records file"
    )
    parser.add_argument(
        "-g", "--road-graph-path",
        type=str,
        required=True,
        help="Path to the road graph file"
    )
    parser.add_argument(
        "-c", "--cameras-info-path",
        type=str,
        required=True,
        help="Path to the cameras information file"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="If True, all GPUs are used for similarity search, otherwise only CPUs"
    )
    args = parser.parse_args()

    random.seed(0)

    run(args.input_path, args.road_graph_path, args.cameras_info_path, args.use_gpu)
