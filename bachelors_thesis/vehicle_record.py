from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from uuid import UUID, uuid4

import networkx as nx
import numpy as np
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

import networking_pb2
from config import DIMENSION
from util import feature_from_base64, normalize, calculate_similarity, edit_distance_gain, clip, get_trace, get_path, \
    get_node_path

RECORD_ID = "record_id"
VEHICLE_ID = "vehicle_id"
CAMERA_ID = "camera_id"
VEHICLE_FEATURE = "car_feature"
LICENSE_PLATE_FEATURE = "plate_feature"
LICENSE_PLATE_TEXT = "plate_text"
TIMESTAMP = "time"


def load_records(record_path: str) -> list[VehicleRecord]:
    records = list()
    with open(record_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            records.append(VehicleRecord.build_record(record))
    return records


class Record(ABC):
    @abstractmethod
    def get_record_id(self) -> UUID:
        pass

    @abstractmethod
    def get_vehicle_id(self) -> int | None:
        pass

    @abstractmethod
    def get_camera_id(self) -> int:
        pass

    @abstractmethod
    def get_timestamp(self) -> int:
        pass

    @abstractmethod
    def is_annotated(self) -> bool:
        pass

    @abstractmethod
    def get_coordinates(self, road_graph: nx.MultiDiGraph, cameras_info: dict) -> (float, float):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class VehicleRecordCompact(Record):
    __record_id: UUID
    __vehicle_id: int | None
    __camera_id: int
    __timestamp: int

    def __init__(self,
                 *,
                 record_id: UUID,
                 vehicle_id: int | None,
                 camera_id: int,
                 timestamp: int):
        self.__record_id = record_id
        self.__vehicle_id = vehicle_id
        self.__camera_id = camera_id
        self.__timestamp = timestamp

    def get_record_id(self) -> UUID:
        return self.__record_id

    def get_vehicle_id(self) -> int | None:
        return self.__vehicle_id

    def get_camera_id(self) -> int:
        return self.__camera_id

    def get_timestamp(self) -> int:
        return self.__timestamp

    def is_annotated(self) -> bool:
        return self.__vehicle_id is not None

    def get_coordinates(self, road_graph: nx.MultiDiGraph, cameras_info: dict) -> (float, float):
        camera = cameras_info[self.__camera_id]
        node_id = camera["node_id"]
        return road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]

    def __eq__(self, other):
        if isinstance(other, VehicleRecordCompact):
            return self.__record_id == other.__record_id
        else:
            return False

    def __hash__(self):
        return hash(self.__record_id)

    @staticmethod
    def from_protobuf(record_pb: networking_pb2.Record) -> VehicleRecordCompact:
        record_id = UUID(record_pb.record_id)
        vehicle_id = record_pb.vehicle_id if record_pb.vehicle_id >= 0 else None

        return VehicleRecordCompact(record_id=record_id,
                                    vehicle_id=vehicle_id,
                                    camera_id=record_pb.camera_id,
                                    timestamp=record_pb.timestamp)


class VehicleRecord(Record):
    __record_id: UUID
    __vehicle_id: int | None
    __camera_id: int
    __vehicle_feature: np.ndarray
    __license_plate_feature: np.ndarray | None
    __license_plate_text: str | None
    __timestamp: int
    __cluster: VehicleRecordCluster | None

    def __init__(self,
                 *,
                 record_id: UUID,
                 vehicle_id: int | None,
                 camera_id: int,
                 vehicle_feature: np.ndarray,
                 license_plate_feature: np.ndarray | None,
                 license_plate_text: str | None,
                 timestamp: int):
        self.__record_id = record_id
        self.__vehicle_id = vehicle_id
        self.__camera_id = camera_id
        self.__vehicle_feature = vehicle_feature
        self.__license_plate_feature = license_plate_feature
        self.__license_plate_text = license_plate_text
        self.__timestamp = timestamp
        self.__cluster = None

    def get_record_id(self) -> UUID:
        return self.__record_id

    def get_vehicle_id(self) -> int | None:
        return self.__vehicle_id

    def get_camera_id(self) -> int:
        return self.__camera_id

    def get_vehicle_feature(self) -> np.ndarray:
        return self.__vehicle_feature

    def get_license_plate_feature(self) -> np.ndarray | None:
        return self.__license_plate_feature

    def get_license_plate_text(self) -> str | None:
        return self.__license_plate_text

    def get_timestamp(self) -> int:
        return self.__timestamp

    def get_cluster(self) -> VehicleRecordCluster:
        return self.__cluster

    def set_cluster(self, cluster: VehicleRecordCluster) -> None:
        self.__cluster = cluster

    def is_annotated(self) -> bool:
        return self.__vehicle_id is not None

    def has_license_plate(self) -> bool:
        return self.__license_plate_feature is not None

    def has_assigned_cluster(self) -> bool:
        return self.__cluster is not None

    def get_coordinates(self, road_graph: nx.MultiDiGraph, cameras_info: dict) -> (float, float):
        camera = cameras_info[self.__camera_id]
        node_id = camera["node_id"]
        return road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]

    def __eq__(self, other):
        if isinstance(other, VehicleRecord):
            return self.__record_id == other.__record_id
        else:
            return False

    def __hash__(self):
        return hash(self.__record_id)

    @staticmethod
    def build_record(record: dict) -> VehicleRecord:
        vehicle_feature = feature_from_base64(record[VEHICLE_FEATURE])
        license_plate_feature = feature_from_base64(record[LICENSE_PLATE_FEATURE])
        license_plate_text = record[LICENSE_PLATE_TEXT] if license_plate_feature is not None else None

        return VehicleRecord(record_id=UUID(record[RECORD_ID]),
                             vehicle_id=record[VEHICLE_ID],
                             camera_id=record[CAMERA_ID],
                             vehicle_feature=vehicle_feature,
                             license_plate_feature=license_plate_feature,
                             license_plate_text=license_plate_text,
                             timestamp=record[TIMESTAMP])

    def to_protobuf(self, record_pb: networking_pb2.Record) -> None:
        record_pb.record_id = self.__record_id.hex
        record_pb.vehicle_id = self.__vehicle_id if self.is_annotated() else -1
        record_pb.camera_id = self.__camera_id
        record_pb.timestamp = self.__timestamp


class Cluster(ABC):
    @abstractmethod
    def get_cluster_id(self) -> UUID:
        pass

    @abstractmethod
    def get_records(self) -> set[Record]:
        pass

    @abstractmethod
    def get_centroid_vehicle_feature(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_centroid_license_plate_feature(self) -> np.ndarray | None:
        pass

    @abstractmethod
    def get_centroid_license_plate_text(self) -> str | None:
        pass

    @abstractmethod
    def get_node_path(self) -> list[int] | None:
        pass

    @abstractmethod
    def set_node_path(self, node_path: list[int]) -> None:
        pass

    @abstractmethod
    def has_node_path(self) -> bool:
        pass

    @abstractmethod
    def has_valid_node_path(self) -> bool:
        pass

    @abstractmethod
    def get_size(self) -> int:
        pass

    @abstractmethod
    def get_ordered_records(self) -> list[Record]:
        pass

    @abstractmethod
    def get_trace(self, road_graph: nx.MultiDiGraph, cameras_info: dict, *, project=True) -> Trace:
        pass

    @abstractmethod
    def has_license_plate(self) -> bool:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class VehicleRecordClusterCompact(Cluster):
    __cluster_id: UUID
    __records: set[Record]
    __centroid_vehicle_feature: np.ndarray
    __centroid_license_plate_feature: np.ndarray | None
    __centroid_license_plate_text: str | None
    __node_path: list[int]

    def __init__(self,
                 *,
                 cluster_id: UUID,
                 records: set[Record],
                 centroid_vehicle_feature: np.ndarray,
                 centroid_license_plate_feature: np.ndarray | None,
                 centroid_license_plate_text: str | None,
                 node_path: list[int]):
        self.__cluster_id = cluster_id
        self.__records = records
        self.__centroid_vehicle_feature = centroid_vehicle_feature
        self.__centroid_license_plate_feature = centroid_license_plate_feature
        self.__centroid_license_plate_text = centroid_license_plate_text
        self.__node_path = node_path

    def get_cluster_id(self) -> UUID:
        return self.__cluster_id

    def get_records(self) -> set[Record]:
        return self.__records

    def get_centroid_vehicle_feature(self) -> np.ndarray:
        return self.__centroid_vehicle_feature

    def get_centroid_license_plate_feature(self) -> np.ndarray | None:
        return self.__centroid_license_plate_feature

    def get_centroid_license_plate_text(self) -> str | None:
        return self.__centroid_license_plate_text

    def get_node_path(self) -> list[int] | None:
        return self.__node_path

    def set_node_path(self, node_path: list[int]) -> None:
        self.__node_path = node_path

    def has_node_path(self) -> bool:
        return self.__node_path is not None

    def has_valid_node_path(self) -> bool:
        return self.has_node_path() and len(self.__node_path) > 1

    def get_size(self) -> int:
        return len(self.__records)

    def get_ordered_records(self) -> list[Record]:
        records = list(self.__records)
        records.sort(key=lambda r: r.get_timestamp())
        return records

    def get_trace(self, road_graph: nx.MultiDiGraph, cameras_info: dict, *, project=True) -> Trace:
        return get_trace(self.get_ordered_records(), road_graph, cameras_info, project=project)

    def has_license_plate(self) -> bool:
        return self.__centroid_license_plate_feature is not None and self.__centroid_license_plate_text is not None

    def __eq__(self, other):
        if isinstance(other, VehicleRecordClusterCompact):
            return self.__cluster_id == other.__cluster_id
        else:
            return False

    def __hash__(self):
        return hash(self.__cluster_id)

    @staticmethod
    def from_protobuf(cluster_pb: networking_pb2.Cluster) -> VehicleRecordClusterCompact:
        records = set()
        for record_pb in cluster_pb.records:
            records.add(VehicleRecordCompact.from_protobuf(record_pb))

        cluster_id = UUID(cluster_pb.cluster_id)
        centroid_vehicle_feature = np.array(cluster_pb.centroid_vehicle_feature).astype(np.float32)

        centroid_license_plate_feature = None
        if len(cluster_pb.centroid_license_plate_feature) != 0:
            centroid_license_plate_feature = np.array(cluster_pb.centroid_license_plate_feature).astype(np.float32)

        centroid_license_plate_text = None
        if cluster_pb.centroid_license_plate_text != "":
            centroid_license_plate_text = cluster_pb.centroid_license_plate_text

        node_path = cluster_pb.node_path

        return VehicleRecordClusterCompact(cluster_id=cluster_id,
                                           centroid_vehicle_feature=centroid_vehicle_feature,
                                           centroid_license_plate_feature=centroid_license_plate_feature,
                                           centroid_license_plate_text=centroid_license_plate_text,
                                           node_path=node_path,
                                           records=records)

    @staticmethod
    def from_clusters(clusters: set[Cluster],
                      road_graph: nx.MultiDiGraph,
                      cameras_info: dict,
                      project=True) -> VehicleRecordClusterCompact:
        cluster_id = uuid4()

        centroid_vehicle_feature = np.zeros(DIMENSION).astype(np.float32)
        for cluster in clusters:
            centroid_vehicle_feature += cluster.get_centroid_vehicle_feature()
        centroid_vehicle_feature /= len(clusters)

        centroid_license_plate_feature = np.zeros(DIMENSION).astype(np.float32)
        centroid_license_plate_text_count = defaultdict(int)
        number_of_clusters_with_license_plates = 0
        for cluster in filter(lambda c: c.has_license_plate(), clusters):
            number_of_clusters_with_license_plates += 1
            centroid_license_plate_feature += cluster.get_centroid_license_plate_feature()
            centroid_license_plate_text_count[cluster.get_centroid_license_plate_text()] += 1
        centroid_license_plate_feature /= number_of_clusters_with_license_plates

        records = list()
        for cluster in clusters:
            for record in cluster.get_records():
                records.append(record)

        records.sort(key=lambda r: r.get_timestamp())

        road_map = NxMap(parse_osmnx_graph(road_graph, xy=project, network_type=NetworkType.DRIVE))
        trace = get_trace(records, road_graph, cameras_info, project=project)
        matcher = LCSSMatcher(road_map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        node_path = None
        if not path_df.empty:
            path = get_path(road_graph, path_df)
            if path is not None and len(path) > 0:
                node_path = get_node_path(path)

        return VehicleRecordClusterCompact(cluster_id=cluster_id,
                                           centroid_vehicle_feature=centroid_vehicle_feature,
                                           centroid_license_plate_feature=centroid_license_plate_feature,
                                           centroid_license_plate_text=max(centroid_license_plate_text_count,
                                                                           key=centroid_license_plate_text_count.get),
                                           records=set(records),
                                           node_path=node_path)


class VehicleRecordCluster(Cluster):
    __cluster_id: UUID
    __records: set[VehicleRecord]
    __centroid_vehicle_feature: np.ndarray
    __number_of_vehicle_features: int
    __centroid_license_plate_feature: np.ndarray
    __number_of_license_plate_features: int
    __license_plate_text_count: dict[str, int]
    __weight_vehicle_similarity: float
    __weight_license_plate_similarity: float
    __path: list[tuple[int, int, int]] | None
    __node_path: list[int] | None

    def __init__(self,
                 *,
                 dimension: int,
                 weight_vehicle_similarity: float,
                 weight_license_plate_similarity: float):
        self.__cluster_id = uuid4()
        self.__records = set()
        self.__centroid_vehicle_feature = np.zeros(dimension, dtype=np.float32)
        self.__number_of_vehicle_features = 0
        self.__centroid_license_plate_feature = np.zeros(dimension, dtype=np.float32)
        self.__number_of_license_plate_features = 0
        self.__license_plate_text_count = defaultdict(int)
        self.__weight_vehicle_similarity = weight_vehicle_similarity
        self.__weight_license_plate_similarity = weight_license_plate_similarity
        self.__path = None
        self.__node_path = None

    def get_cluster_id(self) -> UUID:
        return self.__cluster_id

    def get_records(self) -> set[Record]:
        return self.__records

    def get_centroid_vehicle_feature(self) -> np.ndarray:
        return self.__centroid_vehicle_feature

    def get_centroid_license_plate_feature(self) -> np.ndarray | None:
        return self.__centroid_license_plate_feature

    def get_centroid_license_plate_text(self) -> str | None:
        return max(self.__license_plate_text_count, key=self.__license_plate_text_count.get)

    def get_node_path(self) -> list[int] | None:
        return self.__node_path

    def set_node_path(self, node_path: list[int]) -> None:
        self.__node_path = node_path

    def has_node_path(self) -> bool:
        return self.__node_path is not None

    def has_valid_node_path(self) -> bool:
        return self.has_node_path() and len(self.__node_path) > 1

    def get_size(self) -> int:
        return len(self.__records)

    def get_ordered_records(self) -> list[VehicleRecord]:
        records = list(self.__records)
        records.sort(key=lambda r: r.get_timestamp())
        return records

    def get_trace(self, road_graph: nx.MultiDiGraph, cameras_info: dict, *, project=True) -> Trace:
        return get_trace(self.get_ordered_records(), road_graph, cameras_info, project=project)

    def has_license_plate(self) -> bool:
        return self.__number_of_license_plate_features != 0

    def __eq__(self, other):
        if isinstance(other, VehicleRecordCluster):
            return self.__cluster_id == other.__cluster_id
        else:
            return False

    def __hash__(self):
        return hash(self.__cluster_id)

    def add_record(self, record: VehicleRecord):
        self.__records.add(record)
        record.set_cluster(self)

        self.__number_of_vehicle_features += 1
        if record.has_license_plate():
            self.__number_of_license_plate_features += 1
            self.__license_plate_text_count[record.get_license_plate_text()] += 1

            # Running normalized mean
            self.__centroid_license_plate_feature += (normalize(
                record.get_license_plate_feature()) - self.__centroid_license_plate_feature) / self.__number_of_license_plate_features

        # Running normalized mean
        self.__centroid_vehicle_feature += (normalize(
            record.get_vehicle_feature()) - self.__centroid_vehicle_feature) / self.__number_of_vehicle_features

    def calculate_similarity_to_record(self, record: VehicleRecord) -> float:
        vehicle_similarity = calculate_similarity(self.__centroid_vehicle_feature, record.get_vehicle_feature())
        if self.__number_of_license_plate_features != 0 and record.has_license_plate():
            license_plate_similarity = calculate_similarity(self.__centroid_license_plate_feature,
                                                            record.get_license_plate_feature())
            centroid_license_plate_text = self.get_centroid_license_plate_text()

            return clip(self.__weight_vehicle_similarity * vehicle_similarity
                        + self.__weight_license_plate_similarity * license_plate_similarity
                        + edit_distance_gain(centroid_license_plate_text, record.get_license_plate_text()))
        else:
            return clip(vehicle_similarity)

    def get_path(self) -> list[tuple[int, int, int]] | None:
        return self.__path

    def set_path(self, path: list[tuple[int, int, int]]) -> None:
        self.__path = path

    def has_path(self) -> bool:
        return self.__path is not None

    def has_valid_path(self) -> bool:
        return self.has_path() and len(self.__path) > 0

    def to_protobuf(self, cluster_pb: networking_pb2.Cluster) -> None:
        cluster_pb.cluster_id = self.__cluster_id.hex
        cluster_pb.centroid_vehicle_feature.extend(self.__centroid_vehicle_feature.tolist())

        if self.has_license_plate():
            cluster_pb.centroid_license_plate_feature.extend(self.__centroid_license_plate_feature.tolist())
            cluster_pb.centroid_license_plate_text = self.get_centroid_license_plate_text()

        if self.has_node_path():
            cluster_pb.node_path.extend(self.__node_path)

        for record in self.__records:
            record_pb = cluster_pb.records.add()
            record.to_protobuf(record_pb)
