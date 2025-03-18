import json

from vehicle_record import VehicleRecord


def load_records(paths: list[str]) -> list[VehicleRecord]:
    records = list()
    for path in paths:
        with open(path, mode="r", encoding="utf-8") as file:
            for record_id, line in enumerate(file):
                record = json.loads(line)
                records.append(VehicleRecord(record_id, record))

    return records
