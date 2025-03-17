import base64
import uuid
from uuid import UUID

import numpy as np

VEHICLE_ID = "vehicle_id"
CAMERA_ID = "camera_id"
VEHICLE_FEATURE = "car_feature"
LICENSE_PLATE_FEATURE = "plate_feature"
LICENSE_PLATE_TEXT = "plate_text"
TIMESTAMP = "time"

DIMENSIONS = 256


class VehicleRecord:
    id: UUID
    vehicle_id: int | None
    camera_id: int
    vehicle_feature: np.ndarray[DIMENSIONS]
    license_plate_feature: np.ndarray[DIMENSIONS] | None
    license_plate_text: str | None
    timestamp: int

    def __init__(self, record: dict[str, int | str | None]):
        self.id = uuid.uuid4()
        self.vehicle_id = record[VEHICLE_ID]
        self.camera_id = record[CAMERA_ID]
        self.vehicle_feature = self._feature_from_base64(record[VEHICLE_FEATURE])
        self.license_plate_feature = self._feature_from_base64(record[LICENSE_PLATE_FEATURE])
        self.license_plate_text = record[LICENSE_PLATE_TEXT]
        self.timestamp = record[TIMESTAMP]

    def _feature_from_base64(self, feature: str | None) -> np.ndarray[DIMENSIONS] | None:
        if feature is None:
            return None
        else:
            return np.frombuffer(base64.b64decode(feature), dtype=np.float32)

    def __eq__(self, other):
        return self.id.__eq__(other.id)

    def __hash__(self):
        return self.id.__hash__()
