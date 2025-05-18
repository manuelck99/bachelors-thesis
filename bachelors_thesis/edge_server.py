import logging
from argparse import ArgumentParser, ArgumentTypeError

import zmq

import networking_pb2
from clustering import cluster_records
from map_matching import map_match
from region import load_regions, Region
from util import load, load_graph

logger = logging.getLogger(__name__)


def parse_auxiliary_region(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise ArgumentTypeError(f"{s} can't be parsed as an auxiliary region")
    return int(parts[0]), int(parts[1])


def send_clusters(socket, region: Region) -> None:
    for cluster in filter(lambda c: c.has_valid_node_path(), region.clusters):
        envelope = networking_pb2.Envelope()

        if region.is_auxiliary:
            i, j = region.region_id
            envelope.region_id = f"{i}-{j}"
        else:
            envelope.region_id = str(region.region_id)

        envelope.is_auxiliary = region.is_auxiliary
        cluster.to_protobuf(envelope.cluster)
        socket.send(envelope.SerializeToString())

    envelope = networking_pb2.Envelope()

    if region.is_auxiliary:
        i, j = region.region_id
        envelope.region_id = f"{i}-{j}"
    else:
        envelope.region_id = str(region.region_id)

    envelope.is_auxiliary = region.is_auxiliary
    envelope.done = True
    socket.send(envelope.SerializeToString())


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        region: int,
        auxiliary_regions: list[tuple[int, int]],
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
    map_match(region.clusters, road_graph, cameras_info)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect("tcp://localhost:5555")

    send_clusters(socket, region)

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
        map_match(aux_region.clusters, road_graph, cameras_info)

    for aux_region in aux_regions:
        send_clusters(socket, aux_region)


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
        args.use_gpu)
