from __future__ import annotations

from collections import defaultdict

import numpy as np

import config as cfg
import util

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

    def __init__(self, record_id: int, record: dict[str, int | str | None]):
        self.record_id = record_id
        self.vehicle_id = record[VEHICLE_ID]
        self.camera_id = record[CAMERA_ID]
        self.vehicle_feature = util.feature_from_base64(record[VEHICLE_FEATURE])
        self.license_plate_feature = util.feature_from_base64(record[LICENSE_PLATE_FEATURE])
        self.license_plate_text = record[LICENSE_PLATE_TEXT]
        self.timestamp = record[TIMESTAMP]
        self.cluster = None

        # TODO: Some records have a license plate feature, but it's all zeros, even though apparently there is license plate text
        # if record[LICENSE_PLATE_FEATURE] is not None and not np.any(self.license_plate_feature):
        #     print("-------------------------------------------------------------------------")
        #     print("License plate feature (encoded)", record[LICENSE_PLATE_FEATURE])
        #     print("License plate feature (decoded)", self.license_plate_feature)
        #     print("License plate text ", self.license_plate_text)
        #     print("Record ID ", record_id)
        #     print("Vehicle ID ", self.vehicle_id)

    def has_assigned_cluster(self) -> bool:
        return self.cluster is not None

    def has_license_plate(self) -> bool:
        return self.license_plate_feature is not None

    def __eq__(self, other):
        if isinstance(other, VehicleRecord):
            return self.record_id == other.record_id
        else:
            return False

    def __hash__(self):
        return hash(self.record_id)


class VehicleRecordCluster:
    _records: dict[int, VehicleRecord]
    _centroid_vehicle_feature: np.ndarray
    _number_of_vehicle_features: int
    _centroid_license_plate_feature: np.ndarray
    _number_of_license_plate_features: int
    _license_plate_text_count: dict[str, int]

    def __init__(self):
        self._records = dict()
        self._centroid_vehicle_feature = np.zeros(cfg.DIMENSION, dtype=np.float32)
        self._number_of_vehicle_features = 0
        self._centroid_license_plate_feature = np.zeros(cfg.DIMENSION, dtype=np.float32)
        self._number_of_license_plate_features = 0
        self._license_plate_text_count = defaultdict(int)

    def add_record(self, record: VehicleRecord):
        self._records[record.record_id] = record
        record.cluster = self

        self._number_of_vehicle_features += 1
        if record.has_license_plate():
            self._number_of_license_plate_features += 1
            self._license_plate_text_count[record.license_plate_text] += 1

            # Running normalized mean
            self._centroid_license_plate_feature += (util.normalize(
                record.license_plate_feature) - self._centroid_license_plate_feature) / self._number_of_license_plate_features

        # Running normalized mean
        self._centroid_vehicle_feature += (util.normalize(
            record.vehicle_feature) - self._centroid_vehicle_feature) / self._number_of_vehicle_features

    def calculate_similarity_to_record(self, record: VehicleRecord) -> float:
        vehicle_similarity = util.calculate_similarity(self._centroid_vehicle_feature, record.vehicle_feature)
        if self._number_of_license_plate_features != 0 and record.has_license_plate():
            license_plate_similarity = util.calculate_similarity(self._centroid_license_plate_feature,
                                                                 record.license_plate_feature)
            centroid_license_plate_text = self._calculate_centroid_license_plate_text()

            return util.clip((cfg.WEIGHT_VEHICLE_SIMILARITY * vehicle_similarity
                              + cfg.WEIGHT_LICENSE_PLATE_SIMILARITY * license_plate_similarity
                              + util.edit_distance_gain(centroid_license_plate_text, record.license_plate_text)))
        else:
            return util.clip(vehicle_similarity)

    def size(self) -> int:
        return len(self._records)

    def _calculate_centroid_license_plate_text(self) -> str:
        return max(self._license_plate_text_count, key=self._license_plate_text_count.get)
