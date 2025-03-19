import json

from vehicle_record import VehicleRecord


def load_records(paths: list[str]) -> list[VehicleRecord]:
    count = 0
    records = list()
    for path in paths:
        with open(path, mode="r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                records.append(VehicleRecord(count, record))
                count += 1

    return records
