import time
from argparse import ArgumentParser

from clustering import cluster_records
from evaluation import save_clusters
from map_matching import map_match_clusters
from util import load, load_graph, setup_logger, log_info
from vehicle_record import load_records


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        clusters_output_path: str,
        use_gpu: bool) -> None:
    t0 = time.time_ns()
    records = list(load_records(records_path, transformed=True))
    log_info(f"Number of records: {len(records)}")

    clusters = cluster_records(records, use_gpu=use_gpu)

    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match_clusters(clusters, road_graph, cameras_info)

    save_clusters(clusters, clusters_output_path)
    t1 = time.time_ns()
    log_info(f"Runtime [ms]: {(t1 - t0) / 1000 / 1000}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--records-path",
        type=str,
        required=True,
        help="Path to the records file"
    )
    parser.add_argument(
        "--road-graph-path",
        type=str,
        required=True,
        help="Path to the road graph file"
    )
    parser.add_argument(
        "--cameras-info-path",
        type=str,
        required=True,
        help="Path to the cameras information file"
    )
    parser.add_argument(
        "--clusters-output-path",
        type=str,
        required=True,
        help="Path to the file, where clusters should be saved to"
    )
    parser.add_argument(
        "--logging-path",
        type=str,
        required=True,
        help="Path to the logging file"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use all GPUs for similarity search, otherwise use only CPUs"
    )
    args = parser.parse_args()

    setup_logger(args.logging_path)

    run(args.records_path,
        args.road_graph_path,
        args.cameras_info_path,
        args.clusters_output_path,
        args.use_gpu)
