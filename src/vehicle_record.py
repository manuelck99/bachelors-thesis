from __future__ import annotations

import base64
from typing import Self

import numpy as np

VEHICLE_ID = "vehicle_id"
CAMERA_ID = "camera_id"
VEHICLE_FEATURE = "car_feature"
LICENSE_PLATE_FEATURE = "plate_feature"
LICENSE_PLATE_TEXT = "plate_text"
TIMESTAMP = "time"


class VehicleRecord:
    record_id: int
    vehicle_id: int | None
    camera_id: int
    vehicle_feature: np.ndarray
    license_plate_feature: np.ndarray | None
    license_plate_text: str | None
    timestamp: int
    cluster: VehicleRecordCluster | None
    similarity_to_cluster_centroid: float | None

    def __init__(self, record_id: int, record: dict[str, int | str | None]):
        self.record_id = record_id
        self.vehicle_id = record[VEHICLE_ID]
        self.camera_id = record[CAMERA_ID]
        self.vehicle_feature = self._feature_from_base64(record[VEHICLE_FEATURE])
        self.license_plate_feature = self._feature_from_base64(record[LICENSE_PLATE_FEATURE])
        self.license_plate_text = record[LICENSE_PLATE_TEXT]
        self.timestamp = record[TIMESTAMP]
        self.cluster = None
        self.similarity_to_cluster_centroid = None

    def _feature_from_base64(self, feature: str | None) -> np.ndarray | None:
        if feature is None:
            return None
        else:
            return np.frombuffer(base64.b64decode(feature), dtype=np.float32)

    def __eq__(self, other):
        return self.record_id == other.record_id

    def __hash__(self):
        return hash(self.record_id)


class VehicleRecordCluster:
    _records: dict[int, VehicleRecord]
    _centroid_vehicle_feature: np.ndarray | None
    _centroid_license_plate_feature: np.ndarray | None
    _centroid_license_plate_text: str | None

    def __init__(self, dimension: int):
        self._records = dict()
        self._centroid_vehicle_feature = np.zeros(dimension, dtype=np.float32)
        self._centroid_license_plate_feature = np.zeros(dimension, dtype=np.float32)
        self._centroid_license_plate_text = None

    def _calculate_centroid(self, record: VehicleRecord, added: bool):
        pass

    def add_record(self, record: VehicleRecord):
        self._records[record.record_id] = record
        record.cluster = self
        self._calculate_centroid(record, added=True)

    def remove_record(self, record: VehicleRecord):
        del self._records[record.record_id]
        record.cluster = None
        self._calculate_centroid(record, added=False)

    def calculate_similarity_to_cluster(self, other: Self) -> float:
        pass

    def calculate_similarity_to_record(self, record: VehicleRecord) -> float:
        pass
