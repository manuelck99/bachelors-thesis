from __future__ import annotations

import json

from util import load
from vehicle_record import VehicleRecord, VehicleRecordCluster, VehicleRecordClusterCompact

type RegionID = int | tuple[int, int]


def load_region(records_path: str,
                region_partitioning_path: str,
                region_id: int) -> Region:
    region = Region(region_id=region_id, is_auxiliary=False)

    region_partitioning: dict = load(region_partitioning_path)
    with open(records_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            region_record = VehicleRecord.build_record(record)
            if region.is_record_in_region(region_record, region_partitioning):
                region.add_record(region_record)

    return region


def load_auxiliary_region(records_path: str,
                          region_partitioning_path: str,
                          aux_region_id: tuple[int, int]) -> Region:
    aux_region = Region(region_id=aux_region_id, is_auxiliary=True)

    region_partitioning: dict = load(region_partitioning_path)
    with open(records_path, mode="r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)

            aux_region_record = VehicleRecord.build_record(record)
            if aux_region.is_record_in_region(aux_region_record, region_partitioning):
                aux_region.add_record(aux_region_record)

    return aux_region


class Region:
    region_id: RegionID
    is_auxiliary: bool
    records: list[VehicleRecord]
    clusters: set[VehicleRecordCluster]

    def __init__(self, *, region_id: RegionID, is_auxiliary: bool):
        self.region_id = region_id
        self.is_auxiliary = is_auxiliary
        self.records = list()
        self.clusters = set()

    def add_record(self, record: VehicleRecord) -> None:
        self.records.append(record)

    def is_record_in_region(self, record: VehicleRecord, region_partitioning: dict) -> bool:
        region = region_partitioning[self.region_id]
        camera_ids = region["cameras"]
        return record.get_camera_id() in camera_ids

    def number_of_records(self) -> int:
        return len(self.records)

    def number_of_clusters(self) -> int:
        return len(self.clusters)

    def number_of_singleton_clusters(self) -> int:
        count = 0
        for cluster in self.clusters:
            if cluster.get_size() == 1:
                count += 1
        return count

    def get_name(self) -> str:
        if self.is_auxiliary:
            i, j = self.region_id
            return f"{i}-{j}"
        else:
            return str(self.region_id)

    def __eq__(self, other):
        if isinstance(other, Region):
            return self.is_auxiliary == other.is_auxiliary and self.region_id == other.region_id
        else:
            return False

    def __hash__(self):
        return hash((self.is_auxiliary, self.region_id))


class RegionCompact:
    region_id: RegionID
    is_auxiliary: bool
    clusters: set[VehicleRecordClusterCompact]

    def __init__(self,
                 *,
                 region_id: RegionID,
                 is_auxiliary: bool):
        self.region_id = region_id
        self.is_auxiliary = is_auxiliary
        self.clusters = set()

    def add_cluster(self, cluster: VehicleRecordClusterCompact) -> None:
        self.clusters.add(cluster)

    def number_of_clusters(self) -> int:
        return len(self.clusters)

    def number_of_singleton_clusters(self) -> int:
        count = 0
        for cluster in self.clusters:
            if cluster.get_size() == 1:
                count += 1
        return count

    def __eq__(self, other):
        if isinstance(other, RegionCompact):
            return self.is_auxiliary == other.is_auxiliary and self.region_id == other.region_id
        else:
            return False

    def __hash__(self):
        return hash((self.is_auxiliary, self.region_id))
