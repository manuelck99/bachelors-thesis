import logging
from argparse import ArgumentParser

from clustering import cluster_records
from evaluation import cluster_evaluation_with_record_gt, trajectory_evaluation_with_record_gt, save_vehicle_clusters
from map_matching import map_match_clusters
from util import load, load_graph
from vehicle_record import load_records

logger = logging.getLogger(__name__)


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        clusters_output_path: str,
        use_gpu: bool) -> None:
    records = load_records(records_path)
    logger.info(f"Number of records: {len(records)}")

    # Clustering
    clusters = cluster_records(records, use_gpu=use_gpu)
    singleton_clusters = {cluster for cluster in clusters if cluster.get_size() == 1}
    logger.info(f"Number of clusters: {len(clusters)}")
    logger.info(f"Number of singleton clusters: {len(singleton_clusters)}")
    logger.info(f"Number of non-singleton clusters: {len(clusters) - len(singleton_clusters)}")

    # Map-matching
    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match_clusters(clusters, road_graph, road_graph_path, cameras_info)
    map_matched_clusters = {cluster for cluster in clusters if cluster.has_valid_node_path()}
    logger.info(f"Number of map-matched clusters: {len(clusters) - len(map_matched_clusters)}")

    # Cluster evaluation
    precision, recall, f1_score, expansion = cluster_evaluation_with_record_gt(records, clusters)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")

    # Trajectory evaluation
    lcss, edr, stlc = trajectory_evaluation_with_record_gt(records,
                                                           clusters,
                                                           road_graph,
                                                           road_graph_path,
                                                           cameras_info,
                                                           gamma=0.8,
                                                           epsilon=50)
    logger.info(f"LCSS distance: {lcss}")
    logger.info(f"EDR distance: {edr}")
    logger.info(f"STLC distance: {stlc}")

    save_vehicle_clusters(clusters, clusters_output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

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
        help="Path to the file, where ground-truth vehicle clusters should be saved to"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use all GPUs for similarity search, otherwise use only CPUs"
    )
    args = parser.parse_args()

    run(args.records_path,
        args.road_graph_path,
        args.cameras_info_path,
        args.clusters_output_path,
        args.use_gpu)
