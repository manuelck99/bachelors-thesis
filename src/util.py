import base64
import json

import numpy as np
from Levenshtein import distance

from vehicle_record import VehicleRecord


def load_records(paths: list[str]) -> list[VehicleRecord]:
    record_id = 0
    records = list()
    for path in paths:
        with open(path, mode="r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                records.append(VehicleRecord(record_id, record))
                record_id += 1

    return records


def feature_from_base64(f: str | None) -> np.ndarray | None:
    if f is None:
        return None
    else:
        return np.frombuffer(base64.b64decode(f), dtype=np.float32)


def calculate_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    f = np.dot(f1, f2)
    t = np.linalg.norm(f1) * np.linalg.norm(f2)
    if t == 0:
        # TODO: What to do when one vector is zero?
        return 0.0
    return f / t


def normalize(f: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(f)
    if norm == 0:
        # TODO: What to do when vector is zero?
        return f.copy()
    return f / norm


def clip(v: float, minimum=0.0, maximum=1.0) -> float:
    return max(minimum, min(maximum, v))


def edit_distance_gain(s1: str, s2: str) -> float:
    edit_distance = distance(s1, s2)
    if edit_distance == 0:
        return 0.2
    elif edit_distance == 1:
        return 0.05
    elif edit_distance == 2:
        return 0.0
    elif edit_distance == 3:
        return -0.05
    else:
        return -0.1
