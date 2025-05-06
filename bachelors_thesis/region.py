from __future__ import annotations

import json

from util import feature_from_base64, load
from vehicle_record import VehicleRecord, VehicleRecordCluster, VEHICLE_FEATURE, LICENSE_PLATE_FEATURE


def load_regions(records_path: str,
                 region_partitioning_path: str,
                 region_id: int,
                 aux_region_ids: list[tuple[int, int]]) -> tuple[Region, list[Region]]:
    region = Region(region_id=region_id, is_auxiliary=False)
    aux_regions = [Region(region_id=aux_region_id, is_auxiliary=True) for aux_region_id in aux_region_ids]

    region_partitioning: dict = load(region_partitioning_path)
    with open(records_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            record[VEHICLE_FEATURE] = feature_from_base64(record[VEHICLE_FEATURE])
            record[LICENSE_PLATE_FEATURE] = feature_from_base64(record[LICENSE_PLATE_FEATURE])

            region_record = VehicleRecord(record)
            if region.is_record_in_region(region_record, region_partitioning):
                region.add_record(region_record)

            for aux_region in aux_regions:
                aux_region_record = VehicleRecord(record)
                if aux_region.is_record_in_region(aux_region_record, region_partitioning):
                    aux_region.add_record(aux_region_record)

    return region, aux_regions


class Region:
    region_id: int | tuple[int, int]
    is_auxiliary: bool
    records: list[VehicleRecord]
    clusters: set[VehicleRecordCluster]

    def __init__(self, *, region_id: int | tuple[int, int], is_auxiliary: bool):
        self.region_id = region_id
        self.is_auxiliary = is_auxiliary
        self.records = list()
        self.clusters = set()

    def add_record(self, record: VehicleRecord) -> None:
        self.records.append(record)

    def is_record_in_region(self, record: VehicleRecord, region_partitioning: dict) -> bool:
        region = region_partitioning[self.region_id]
        camera_ids = region["cameras"]
        return record.camera_id in camera_ids

    def number_of_records(self) -> int:
        return len(self.records)

    def number_of_clusters(self) -> int:
        return len(self.clusters)

    def number_of_singleton_clusters(self) -> int:
        count = 0
        for cluster in self.clusters:
            if cluster.size() == 1:
                count += 1
        return count

    def __eq__(self, other):
        if isinstance(other, Region):
            return self.is_auxiliary == other.is_auxiliary and self.region_id == other.region_id
        else:
            return False

    def __hash__(self):
        return hash((self.is_auxiliary, self.region_id))
