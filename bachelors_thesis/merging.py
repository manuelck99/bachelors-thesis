from uuid import UUID

from region import RegionID, RegionCompact
from vehicle_record import Cluster


def find_clusters_to_merge(aux_region: RegionCompact,
                           regions: dict[RegionID, RegionCompact],
                           region_partitioning: dict) -> set[tuple[UUID, UUID]]:
    i, j = aux_region.region_id

    aux_clusters_from_i_to_j = set()
    aux_clusters_from_j_to_i = set()
    for cluster in aux_region.clusters:
        if cluster_crosses_from_i_to_j(cluster, region_partitioning[(i, j)]["edges_i_to_j"]):
            aux_clusters_from_i_to_j.add(cluster)

        if cluster_crosses_from_i_to_j(cluster, region_partitioning[(i, j)]["edges_j_to_i"]):
            aux_clusters_from_j_to_i.add(cluster)

    clusters_to_merge = find_clusters_crossing_from_i_to_j_to_merge(i,
                                                                    j,
                                                                    aux_clusters_from_i_to_j,
                                                                    regions,
                                                                    region_partitioning)
    clusters_to_merge.update(find_clusters_crossing_from_i_to_j_to_merge(j,
                                                                         i,
                                                                         aux_clusters_from_j_to_i,
                                                                         regions,
                                                                         region_partitioning))

    return clusters_to_merge


def cluster_crosses_from_i_to_j(cluster: Cluster,
                                edges_from_i_to_j: set[tuple[int, int]]) -> bool:
    for u, v in zip(cluster.get_node_path(), cluster.get_node_path()[1:]):
        if (u, v) in edges_from_i_to_j:
            return True
    return False


def find_clusters_crossing_from_i_to_j_to_merge(i: int,
                                                j: int,
                                                aux_clusters_from_i_to_j: set[Cluster],
                                                region: dict[RegionID, RegionCompact],
                                                region_partitioning: dict) -> set[tuple[UUID, UUID]]:
    clusters_starting_outside_ending_inside_ij = set()
    for cluster in region[i].clusters:
        if cluster_starts_outside_ends_inside_aux_region(cluster, region_partitioning[(i, j)]["nodes"]):
            clusters_starting_outside_ending_inside_ij.add(cluster)

    clusters_starting_inside_ending_outside_ij = set()
    for cluster in region[j].clusters:
        if cluster_starts_inside_ends_outside_aux_region(cluster, region_partitioning[(i, j)]["nodes"]):
            clusters_starting_inside_ending_outside_ij.add(cluster)

    # TODO: Top K similarity search


def cluster_starts_outside_ends_inside_aux_region(cluster: Cluster,
                                                  aux_region_nodes: set[int]) -> bool:
    node_path = cluster.get_node_path()
    start_node = node_path[0]
    end_node = node_path[-1]
    return start_node not in aux_region_nodes and end_node in aux_region_nodes


def cluster_starts_inside_ends_outside_aux_region(cluster: Cluster,
                                                  aux_region_nodes: set[int]) -> bool:
    node_path = cluster.get_node_path()
    start_node = node_path[0]
    end_node = node_path[-1]
    return start_node in aux_region_nodes and end_node not in aux_region_nodes
