import logging
from argparse import ArgumentParser

from clustering import cluster_records
from evaluation import yu_ao_yan_cluster_evaluation, su_liu_zheng_trajectory_evaluation
from map_matching import map_match
from util import load, load_graph
from vehicle_record import load_records

logger = logging.getLogger(__name__)


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        map_match_proj_graph: bool,
        use_gpu: bool) -> None:
    records = load_records(records_path)
    logger.info(f"Number of records: {len(records)}")

    # Clustering
    clusters = cluster_records(records, use_gpu=use_gpu)
    singleton_clusters = {cluster for cluster in clusters if cluster.size() == 1}
    logger.info(f"Number of clusters: {len(clusters)}")
    logger.info(f"Number of singleton clusters: {len(singleton_clusters)}")
    logger.info(f"Number of non-singleton clusters: {len(clusters) - len(singleton_clusters)}")

    # Map-matching
    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match(clusters, road_graph, cameras_info, project=map_match_proj_graph)

    # Cluster evaluation
    precision, recall, f1_score, expansion = yu_ao_yan_cluster_evaluation(records, clusters)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")

    # Trajectory evaluation
    lcss, edr, stlc = su_liu_zheng_trajectory_evaluation(records,
                                                         clusters,
                                                         road_graph,
                                                         cameras_info,
                                                         project=map_match_proj_graph)
    logger.info(f"LCSS distance: {lcss}")
    logger.info(f"EDR distance: {edr}")
    logger.info(f"STLC distance: {stlc}")


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
        "--map-match-proj-graph",
        action="store_true",
        help="Use a projected graph for map matching"
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
        args.map_match_proj_graph,
        args.use_gpu)
