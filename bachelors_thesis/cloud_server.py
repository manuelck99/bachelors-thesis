import logging
from argparse import ArgumentParser

import zmq
from rasterio.crs import defaultdict

import networking_pb2
from record import Cluster, Record
from util import load

logger = logging.getLogger(__name__)


def parse_region_id(region_id: str, is_auxiliary: bool) -> int | tuple[int, int]:
    if is_auxiliary:
        region_id_parts = region_id.split("-")
        return int(region_id_parts[0]), int(region_id_parts[1])
    else:
        return int(region_id)


def protobuf_to_cluster(cluster_pb) -> Cluster:
    records = list()
    for record in cluster_pb.records:
        records.append(Record(record_id=record.record_id,
                              vehicle_id=record.vehicle_id,
                              camera_id=record.camera_id,
                              timestamp=record.timestamp))

    return Cluster(cluster_id=cluster_pb.cluster_id,
                   centroid_vehicle_feature=cluster_pb.centroid_vehicle_feature,
                   centroid_license_plate_feature=cluster_pb.centroid_license_plate_feature,
                   centroid_license_plate_text=cluster_pb.centroid_license_plate_text,
                   node_path=cluster_pb.node_path,
                   records=records)


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        map_match_proj_graph: bool,
        use_gpu: bool) -> None:
    region_partitioning: dict = load(region_partitioning_path)
    regions_done = dict()
    region_clusters = defaultdict(set)
    for region_id in region_partitioning.keys():
        regions_done[region_id] = False

    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://localhost:5555")
    while not all(regions_done.values()):
        message = socket.recv()
        envelope = networking_pb2.Envelope()
        envelope.ParseFromString(message)

        is_auxiliary: bool = envelope.is_auxiliary
        region_id = parse_region_id(envelope.region_id, is_auxiliary)
        if envelope.WhichOneof("content") == "done":
            regions_done[region_id] = True
        else:
            cluster = protobuf_to_cluster(envelope.cluster)
            region_clusters[region_id].add(cluster)

        print("regions_done: ", regions_done)
        print("region_clusters: ", region_clusters)

    # TODO: Shutdown ZeroMQ


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
        args.map_match_proj_graph,
        args.use_gpu)
