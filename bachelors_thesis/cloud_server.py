import logging
from argparse import ArgumentParser

import zmq

import networking_pb2
from evaluation import load_vehicle_clusters, cluster_evaluation_with_centralized_gt, \
    trajectory_evaluation_with_centralized_gt
from merging import find_clusters_to_merge, merge_clusters
from region import RegionID, RegionCompact
from util import load, load_graph
from vehicle_record import VehicleRecordClusterCompact

logger = logging.getLogger(__name__)


def parse_region_id(region_id: str, is_auxiliary: bool) -> RegionID:
    if is_auxiliary:
        region_id_parts = region_id.split("-")
        return int(region_id_parts[0]), int(region_id_parts[1])
    else:
        return int(region_id)


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        clusters_input_path: str,
        use_gpu: bool) -> None:
    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    region_partitioning: dict = load(region_partitioning_path)

    regions_done = dict()
    regions = dict()
    for region_id, region in region_partitioning.items():
        regions_done[region_id] = False
        if region["auxiliary"]:
            regions[region_id] = RegionCompact(region_id=region_id, is_auxiliary=True)
        else:
            regions[region_id] = RegionCompact(region_id=region_id, is_auxiliary=False)

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://localhost:5555")

    while not all(regions_done.values()):
        message = socket.recv()
        envelope = networking_pb2.Envelope()
        envelope.ParseFromString(message)

        is_auxiliary: bool = envelope.is_auxiliary
        region_id = parse_region_id(envelope.region_id, is_auxiliary)
        if envelope.WhichOneof("content") == "done" and envelope.done:
            regions_done[region_id] = True
        else:
            cluster = VehicleRecordClusterCompact.from_protobuf(envelope.cluster)
            regions[region_id].add_cluster(cluster)

    clusters_to_merge = set()
    for region in regions.values():
        if region.is_auxiliary:
            clusters_to_merge.update(find_clusters_to_merge(region,
                                                            regions,
                                                            region_partitioning=region_partitioning,
                                                            cameras_info=cameras_info,
                                                            use_gpu=use_gpu))

    clusters = dict()
    for region in regions.values():
        for cluster in region.clusters:
            clusters[cluster.get_cluster_id()] = cluster

    clusters = merge_clusters(clusters,
                              clusters_to_merge,
                              road_graph=road_graph,
                              cameras_info=cameras_info)

    clusters = set(clusters.values())
    records = list()
    for cluster in clusters:
        records.extend(cluster.get_records())

    clusters_gt = load_vehicle_clusters(clusters_input_path)

    precision, recall, f1_score, expansion = cluster_evaluation_with_centralized_gt(clusters_gt, clusters)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Expansion: {expansion}")

    # Trajectory evaluation
    lcss, edr, stlc = trajectory_evaluation_with_centralized_gt(clusters_gt,
                                                                clusters,
                                                                road_graph,
                                                                cameras_info,
                                                                gamma=0.8,
                                                                epsilon=50)
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
        "--clusters-input-path",
        type=str,
        required=True,
        help="Path to the file, where ground-truth vehicle clusters should be loaded from"
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
        args.clusters_input_path,
        args.use_gpu)
