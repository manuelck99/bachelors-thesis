import logging
from argparse import ArgumentParser, ArgumentTypeError

from clustering import cluster_records
from evaluation import yu_ao_yan_cluster_evaluation, su_liu_zheng_trajectory_evaluation
from map_matching import map_match
from region import load_regions
from util import load, load_graph

logger = logging.getLogger(__name__)


def parse_auxiliary_region(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise ArgumentTypeError(f"{s} can't be parsed as an auxiliary region")
    return int(parts[0]), int(parts[1])


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        region: int,
        auxiliary_regions: list[tuple[int, int]],
        map_match_proj_graph: bool,
        use_gpu: bool) -> None:
    region, aux_regions = load_regions(records_path,
                                       region_partitioning_path,
                                       region,
                                       auxiliary_regions)
    # Region
    logger.info(f"Number of region {region.region_id} records: {region.number_of_records()}")

    clusters = cluster_records(region.records, use_gpu=use_gpu)
    region.clusters = clusters

    logger.info(f"Region {region.region_id}:")
    logger.info(f"Number of clusters: {region.number_of_clusters()}")
    logger.info(f"Number of singleton clusters: {region.number_of_singleton_clusters()}")
    logger.info(
        f"Number of non-singleton clusters: {region.number_of_clusters() - region.number_of_singleton_clusters()}")

    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match(region.clusters, road_graph, cameras_info, project=map_match_proj_graph)

    # Auxiliary regions
    for aux_region in aux_regions:
        logger.info(f"Number of auxiliary region {aux_region.region_id} records: {aux_region.number_of_records()}")

    for aux_region in aux_regions:
        clusters = cluster_records(aux_region.records, use_gpu=use_gpu)
        aux_region.clusters = clusters

    for aux_region in aux_regions:
        logger.info(f"Auxiliary region {aux_region.region_id}:")
        logger.info(f"Number of clusters: {aux_region.number_of_clusters()}")
        logger.info(f"Number of singleton clusters: {aux_region.number_of_singleton_clusters()}")
        logger.info(
            f"Number of non-singleton clusters: {aux_region.number_of_clusters() - aux_region.number_of_singleton_clusters()}")

    for aux_region in aux_regions:
        map_match(aux_region.clusters, road_graph, cameras_info, project=map_match_proj_graph)

    # Cluster evaluation
    precision, recall, f1_score, expansion = yu_ao_yan_cluster_evaluation(region.records, region.clusters)
    logger.info(f"Region {region.region_id}:")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")

    for aux_region in aux_regions:
        precision, recall, f1_score, expansion = yu_ao_yan_cluster_evaluation(aux_region.records, aux_region.clusters)
        logger.info(f"Auxiliary region {aux_region.region_id}:")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1-Score: {f1_score}")
        logger.info(f"Expansion: {expansion}")

    # Trajectory evaluation
    lcss, edr, stlc = su_liu_zheng_trajectory_evaluation(region.records,
                                                         region.clusters,
                                                         road_graph,
                                                         cameras_info,
                                                         project=map_match_proj_graph)
    logger.info(f"Region {region.region_id}:")
    logger.info(f"LCSS distance: {lcss}")
    logger.info(f"EDR distance: {edr}")
    logger.info(f"STLC distance: {stlc}")

    for aux_region in aux_regions:
        lcss, edr, stlc = su_liu_zheng_trajectory_evaluation(aux_region.records,
                                                             aux_region.clusters,
                                                             road_graph,
                                                             cameras_info,
                                                             project=map_match_proj_graph)
        logger.info(f"Auxiliary region {aux_region.region_id}:")
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
        "--region-partitioning-path",
        type=str,
        required=True,
        help="Path to the region partitioning file"
    )
    parser.add_argument(
        "--region",
        type=int,
        required=True,
        help="ID of the region this edge server should handle"
    )
    parser.add_argument(
        "--auxiliary-regions",
        type=parse_auxiliary_region,
        required=False,
        default=[],
        nargs="*",
        help="IDs of the auxiliary regions this edge server should handle, format \\d+-\\d+"
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
        args.region_partitioning_path,
        args.region,
        args.auxiliary_regions,
        args.map_match_proj_graph,
        args.use_gpu)
