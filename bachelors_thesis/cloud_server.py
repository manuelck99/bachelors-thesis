import time
from argparse import ArgumentParser

import zmq

import networking_pb2
from evaluation import save_clusters
from merging import find_clusters_to_merge, merge_clusters
from region import RegionID, RegionCompact
from util import load, load_graph, setup_logger, log_info
from vehicle_record import VehicleRecordClusterCompact


def parse_region_id(region_id: str, is_auxiliary: bool) -> RegionID:
    if is_auxiliary:
        region_id_parts = region_id.split("-")
        return int(region_id_parts[0]), int(region_id_parts[1])
    else:
        return int(region_id)


def run(road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        clusters_output_path: str,
        socket_address: str,
        use_gpu: bool) -> None:
    t0 = time.time_ns()
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
    socket.bind(socket_address)

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
    save_clusters(clusters, clusters_output_path)
    t1 = time.time_ns()
    log_info(f"Runtime [ms]: {(t1 - t0) / 1000 / 1000}")


if __name__ == "__main__":
    parser = ArgumentParser()
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
        help="Address the socket should bind to"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use all GPUs for similarity search, otherwise use only CPUs"
    )
    args = parser.parse_args()

    setup_logger(args.logging_path)

    run(args.road_graph_path,
        args.cameras_info_path,
        args.region_partitioning_path,
        args.clusters_output_path,
        args.socket_address,
        args.use_gpu)
