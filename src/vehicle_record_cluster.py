from typing import Iterable, Self

import numpy as np

from vehicle_record import VehicleRecord


class VehicleRecordCluster:
    records: set[VehicleRecord]
    vehicle_feature: np.ndarray[256] | None
    license_plate_feature: np.ndarray[256] | None
    license_plate_text: str | None

    def __init__(self, records: Iterable[VehicleRecord]):
        self.records = set(records)
        self.vehicle_feature = None
        self.license_plate_feature = None
        self.license_plate_text = None

    def _calculate_centroid(self):
        pass

    def add_record(self, record: VehicleRecord):
        pass

    def remove_record(self, record: VehicleRecord):
        pass

    def calculate_similarity_to(self, other: Self) -> float:
        pass
