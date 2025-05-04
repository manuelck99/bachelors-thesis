from argparse import ArgumentParser, ArgumentTypeError

from region import load_regions


def parse_auxiliary_region(s: str) -> tuple[int, int]:
    parts = s.split("-")
    if len(parts) != 2:
        raise ArgumentTypeError(f"{s} can't be parsed as an auxiliary region")
    return int(parts[0]), int(parts[1])


def run(records_path: str,
        road_graph_path: str,
        cameras_info_path: str,
        region_partitioning_path: str,
        region: int,
        auxiliary_regions: list[tuple[int, int]],
        map_match_proj_graph: bool,
        use_gpu: bool) -> None:
    region, aux_regions = load_regions(records_path,
                                       cameras_info_path,
                                       region_partitioning_path,
                                       region,
                                       auxiliary_regions)
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--records-path",
        type=str,
        required=True,
        help="Path to the records file"
    )
    parser.add_argument(
        "--road-graph-path",
        type=str,
        required=True,
        help="Path to the road graph file"
    )
    parser.add_argument(
        "--cameras-info-path",
        type=str,
        required=True,
        help="Path to the cameras information file"
    )
    parser.add_argument(
        "--region-partitioning-path",
        type=str,
        required=True,
        help="Path to the region partitioning file"
    )
    parser.add_argument(
        "--region",
        type=int,
        required=True,
        help="ID of the region this edge server should handle"
    )
    parser.add_argument(
        "--auxiliary-regions",
        type=parse_auxiliary_region,
        required=False,
        default=[],
        nargs="*",
        help="IDs of the auxiliary regions this edge server should handle, format \d+-\d+"
    )
    parser.add_argument(
        "--map-match-proj-graph",
        action="store_true",
        help="Use a projected graph for map matching"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use all GPUs for similarity search, otherwise use only CPUs"
    )
    args = parser.parse_args()

    print(args)
