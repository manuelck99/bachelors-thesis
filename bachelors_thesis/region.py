from __future__ import annotations

from vehicle_record import VehicleRecord


def load_regions(records_path: str,
                 cameras_info_path: str,
                 region_partitioning_path: str,
                 region_id: int,
                 aux_region_ids: list[tuple[int, int]]) -> tuple[Region, list[AuxiliaryRegion]]:
    region = Region(region_id=region_id)
    # TODO


class Region:
    region_id: int
    records: list[VehicleRecord]
    neighbour_regions: set[Region]
    adjacent_aux_regions: set[AuxiliaryRegion]

    def __init__(self, *, region_id: int):
        self.region_id = region_id


class AuxiliaryRegion:
    region_id: tuple[int, int]
    records: list[VehicleRecord]
    adjacent_regions: set[Region]

    def __init__(self, *, region_id: tuple[int, int]):
        self.region_id = region_id
