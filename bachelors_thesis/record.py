from uuid import UUID

import numpy as np


class Record:
    record_id: UUID
    vehicle_id: int | None
    camera_id: int
    timestamp: int

    def __init__(self,
                 *,
                 record_id: str,
                 vehicle_id: int,
                 camera_id: int,
                 timestamp: int):
        self.record_id = UUID(record_id)
        self.vehicle_id = vehicle_id if vehicle_id >= 0 else None
        self.camera_id = camera_id
        self.timestamp = timestamp

    def __eq__(self, other):
        if isinstance(other, Record):
            return self.record_id == other.record_id
        else:
            return False

    def __hash__(self):
        return hash(self.record_id)


class Cluster:
    cluster_id: UUID
    centroid_vehicle_feature: np.ndarray
    centroid_license_plate_feature: np.ndarray | None
    centroid_license_plate_text: str | None
    node_path: list[int]
    records: list[Record]

    def __init__(self,
                 *,
                 cluster_id: str,
                 centroid_vehicle_feature: list[float],
                 centroid_license_plate_feature: list[float],
                 centroid_license_plate_text: str,
                 node_path: list[int],
                 records: list[Record]):
        self.cluster_id = UUID(cluster_id)
        self.centroid_vehicle_feature = np.array(centroid_vehicle_feature)
        self.centroid_license_plate_feature = np.array(centroid_license_plate_feature) if len(
            centroid_license_plate_feature) != 0 else None
        self.centroid_license_plate_text = centroid_license_plate_text if centroid_license_plate_text != "" else None
        self.node_path = node_path
        self.records = records

    def __eq__(self, other):
        if isinstance(other, Cluster):
            return self.cluster_id == other.cluster_id
        else:
            return False

    def __hash__(self):
        return hash(self.cluster_id)
