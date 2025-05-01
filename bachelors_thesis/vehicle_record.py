from __future__ import annotations

import json
from collections import defaultdict

import networkx as nx
import numpy as np
from mappymatch.constructs.trace import Trace

from config import DIMENSION, WEIGHT_VEHICLE_SIMILARITY, WEIGHT_LICENSE_PLATE_SIMILARITY
from util import feature_from_base64, normalize, calculate_similarity, edit_distance_gain, clip, get_trace

RECORD_ID = "record_id"
VEHICLE_ID = "vehicle_id"
CAMERA_ID = "camera_id"
VEHICLE_FEATURE = "car_feature"
LICENSE_PLATE_FEATURE = "plate_feature"
LICENSE_PLATE_TEXT = "plate_text"
TIMESTAMP = "time"


def load_records(record_path: str) -> list[VehicleRecord]:
    record_id = 0
    records = list()
    with open(record_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            record[RECORD_ID] = record_id
            record[VEHICLE_FEATURE] = feature_from_base64(record[VEHICLE_FEATURE])
            record[LICENSE_PLATE_FEATURE] = feature_from_base64(record[LICENSE_PLATE_FEATURE])
            records.append(VehicleRecord(record))
            record_id += 1
    return records


class VehicleRecord:
    record_id: int
    vehicle_id: int | None
    camera_id: int
    vehicle_feature: np.ndarray
    license_plate_feature: np.ndarray | None
    license_plate_text: str | None
    timestamp: int
    cluster: VehicleRecordCluster | None

    def __init__(self, record: dict):
        self.record_id = record[RECORD_ID]
        self.vehicle_id = record[VEHICLE_ID]
        self.camera_id = record[CAMERA_ID]
        self.vehicle_feature = record[VEHICLE_FEATURE]
        self.license_plate_feature = record[LICENSE_PLATE_FEATURE]
        # Some records have a license plate feature, but it's all zeros,
        # even though apparently there is license plate text, so the
        # license plate text is also set to None in that case.
        self.license_plate_text = record[LICENSE_PLATE_TEXT] if self.license_plate_feature is not None else None
        self.timestamp = record[TIMESTAMP]
        self.cluster = None

    def has_assigned_cluster(self) -> bool:
        return self.cluster is not None

    def has_license_plate(self) -> bool:
        return self.license_plate_feature is not None

    def is_annotated(self) -> bool:
        return self.vehicle_id is not None

    def get_coordinates(self, road_graph: nx.MultiDiGraph, cameras_info: dict) -> (float, float):
        camera = cameras_info[self.camera_id]
        node_id = camera["node_id"]
        return road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]

    def __eq__(self, other):
        if isinstance(other, VehicleRecord):
            return self.record_id == other.record_id
        else:
            return False

    def __hash__(self):
        return hash(self.record_id)


class VehicleRecordCluster:
    records: dict[int, VehicleRecord]
    centroid_vehicle_feature: np.ndarray
    number_of_vehicle_features: int
    centroid_license_plate_feature: np.ndarray
    number_of_license_plate_features: int
    license_plate_text_count: dict[str, int]
    weight_vehicle_similarity: float
    weight_license_plate_similarity: float
    path: list[tuple[int, int, int]] | None
    node_path: list[int] | None

    def __init__(self, *, dimension: int = DIMENSION,
                 weight_vehicle_similarity: float = WEIGHT_VEHICLE_SIMILARITY,
                 weight_license_plate_similarity: float = WEIGHT_LICENSE_PLATE_SIMILARITY):
        self.records = dict()
        self.centroid_vehicle_feature = np.zeros(dimension, dtype=np.float32)
        self.number_of_vehicle_features = 0
        self.centroid_license_plate_feature = np.zeros(dimension, dtype=np.float32)
        self.number_of_license_plate_features = 0
        self.license_plate_text_count = defaultdict(int)
        self.weight_vehicle_similarity = weight_vehicle_similarity
        self.weight_license_plate_similarity = weight_license_plate_similarity
        self.path = None
        self.node_path = None

    def add_record(self, record: VehicleRecord):
        self.records[record.record_id] = record
        record.cluster = self

        self.number_of_vehicle_features += 1
        if record.has_license_plate():
            self.number_of_license_plate_features += 1
            self.license_plate_text_count[record.license_plate_text] += 1

            # Running normalized mean
            self.centroid_license_plate_feature += (normalize(
                record.license_plate_feature) - self.centroid_license_plate_feature) / self.number_of_license_plate_features

        # Running normalized mean
        self.centroid_vehicle_feature += (normalize(
            record.vehicle_feature) - self.centroid_vehicle_feature) / self.number_of_vehicle_features

    def calculate_similarity_to_record(self, record: VehicleRecord) -> float:
        vehicle_similarity = calculate_similarity(self.centroid_vehicle_feature, record.vehicle_feature)
        if self.number_of_license_plate_features != 0 and record.has_license_plate():
            license_plate_similarity = calculate_similarity(self.centroid_license_plate_feature,
                                                            record.license_plate_feature)
            centroid_license_plate_text = self._calculate_centroid_license_plate_text()

            return clip(self.weight_vehicle_similarity * vehicle_similarity
                        + self.weight_license_plate_similarity * license_plate_similarity
                        + edit_distance_gain(centroid_license_plate_text, record.license_plate_text))
        else:
            return clip(vehicle_similarity)

    def _calculate_centroid_license_plate_text(self) -> str:
        return max(self.license_plate_text_count, key=self.license_plate_text_count.get)

    def size(self) -> int:
        return len(self.records)

    def get_ordered_records(self) -> list[VehicleRecord]:
        records = list(self.records.values())
        records.sort(key=lambda r: r.timestamp)
        return records

    def get_trace(self, road_graph: nx.MultiDiGraph, cameras_info: dict, *, project=True) -> Trace:
        return get_trace(self.get_ordered_records(), road_graph, cameras_info, project=project)

    def has_path(self) -> bool:
        return self.path is not None

    def has_valid_path(self) -> bool:
        return self.has_path() and len(self.path) > 0
