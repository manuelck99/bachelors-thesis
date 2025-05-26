import time
from argparse import ArgumentParser, ArgumentTypeError
from multiprocessing import Process, Lock

import zmq

import networking_pb2
from clustering import cluster_records
from evaluation import save_clusters
from map_matching import map_match_clusters
from region import Region, load_region, load_auxiliary_region
from util import load_graph, load, setup_logger, log_info


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


def process_region(records_path: str,
                   road_graph_path: str,
                   cameras_info_path: str,
                   region_partitioning_path: str,
                   clusters_output_path: str,
                   socket_address: str,
                   region_id: int,
                   lock: Lock) -> None:
    region = load_region(records_path,
                         region_partitioning_path,
                         region_id)
    log_info(f"Number of records: {region.number_of_records()}", region=region.get_name(), lock=lock)

    clusters = cluster_records(region.records, region=region.get_name(), lock=lock)
    region.clusters = clusters

    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match_clusters(region.clusters, road_graph, cameras_info, region=region.get_name(), lock=lock)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(socket_address)
    send_clusters(socket, region)

    save_clusters(region.clusters, clusters_output_path)


def process_auxiliary_region(records_path: str,
                             road_graph_path: str,
                             cameras_info_path: str,
                             region_partitioning_path: str,
                             socket_address: str,
                             aux_region_id: tuple[int, int],
                             lock: Lock) -> None:
    aux_region = load_auxiliary_region(records_path,
                                       region_partitioning_path,
                                       aux_region_id)
    log_info(f"Number of records: {aux_region.number_of_records()}", region=aux_region.get_name(), lock=lock)

    clusters = cluster_records(aux_region.records, region=aux_region.get_name(), lock=lock)
    aux_region.clusters = clusters

    road_graph = load_graph(road_graph_path)
    cameras_info: dict = load(cameras_info_path)
    map_match_clusters(aux_region.clusters, road_graph, cameras_info, region=aux_region.get_name(), lock=lock)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(socket_address)
    send_clusters(socket, aux_region)


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        clusters_output_path: str,
        socket_address: str,
        region: int,
        aux_regions: list[tuple[int, int]]) -> None:
    t0 = time.time_ns()
    lock = Lock()
    processes = list()
    processes.append(Process(target=process_region,
                             args=(records_path,
                                   road_graph_path,
                                   cameras_info_path,
                                   region_partitioning_path,
                                   clusters_output_path,
                                   socket_address,
                                   region,
                                   lock)))

    for aux_region in aux_regions:
        processes.append(Process(target=process_auxiliary_region,
                                 args=(records_path,
                                       road_graph_path,
                                       cameras_info_path,
                                       region_partitioning_path,
                                       socket_address,
                                       aux_region,
                                       lock)))

    for process in processes:
        process.start()

    for process in processes:
        process.join()

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
        "--region-partitioning-path",
        type=str,
        required=True,
        help="Path to the region partitioning file"
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
        "--socket-address",
        type=str,
        required=True,
        help="Address the socket should connect to"
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
    args = parser.parse_args()

    setup_logger(args.logging_path)

    run(args.records_path,
        args.road_graph_path,
        args.cameras_info_path,
        args.region_partitioning_path,
        args.clusters_output_path,
        args.socket_address,
        args.region,
        args.auxiliary_regions)
