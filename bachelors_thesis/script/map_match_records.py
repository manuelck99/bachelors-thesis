"""
For each vehicle identity:
1. Find all their annotated records
2. Order them by time ascending
3. Look up each record's camera ID
4. Match the camera ID to a node ID
5. Match the node ID to node coordinates
6. Create a mappymatch Trace with the record's points
7. Map match the Trace to the road graph
8. Retrieve the path (nodes, edges) for the matching
9. Save visualization of paths
10. Save all paths as ground truths
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from mappymatch.constructs.trace import Trace
from mappymatch.maps.nx.nx_map import NxMap
from mappymatch.maps.nx.readers.osm_readers import parse_osmnx_graph, NetworkType
from mappymatch.matchers.lcss.lcss import LCSSMatcher

sys.path.append("..")

import util

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
RECORDS_PATH = os.path.join(DATA_PATH, "./dataset/records")

NUMBER_OF_ANNOTATED_VEHICLES = 173


def get_coordinates_of_record(record: dict, cameras_info: dict, road_graph: nx.MultiDiGraph) -> (float, float):
    camera_id = record["camera_id"]
    camera = cameras_info[camera_id]
    node_id = camera["node_id"]
    return road_graph.nodes[node_id]["x"], road_graph.nodes[node_id]["y"]


def get_path(path_df: pd.DataFrame) -> (list[int], list[tuple[int, int, int]]):
    nodes = list()
    edges = list()

    for _, (o, d, k) in path_df[["origin_junction_id", "destination_junction_id", "road_key"]].iterrows():
        edges.append((o, d, k))

    for o, d, _ in edges:
        nodes.append(o)
    nodes.append(d)

    return nodes, edges


def is_path_valid(edges, road_graph: nx.MultiDiGraph) -> int:
    for o, d, k in edges:
        if not road_graph.has_edge(o, d, k):
            return 0

    prev_d = edges[0][1]
    for i, (o, d, _) in enumerate(edges[1:]):
        if prev_d != o:
            return i + 1
        prev_d = d

    return 0


if __name__ == "__main__":
    road_graph = util.load_graph(f"{DATA_PATH}/road_graph/road_graph_ox_sim_con_35_nsl_sc.pickle")
    road_graph_og = util.load_graph(f"{DATA_PATH}/road_graph/road_graph_ox_nsl.pickle")
    cameras_info = util.load(f"{DATA_PATH}/road_graph/road_graph_ox_sim_con_35_nsl_sc_cameras.pickle")

    trajectories = list()
    for vehicle_id in range(NUMBER_OF_ANNOTATED_VEHICLES):
        records = list()

        # 1.
        with open(f"{RECORDS_PATH}/vehicles/records-vehicle-{vehicle_id}.json",
                  mode="r",
                  encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                del record["car_feature"]
                del record["plate_feature"]
                del record["plate_text"]
                records.append(record)

        # 2.
        records.sort(key=lambda r: r["time"])

        # 3., 4., 5.
        for record in records:
            lon, lat = get_coordinates_of_record(record, cameras_info, road_graph)
            record["x"] = lon
            record["y"] = lat

        # 6.
        trace = [[record["x"], record["y"]] for record in records]
        trace_df = pd.DataFrame(trace, columns=["longitude", "latitude"])
        trajectories.append({"vehicle_id": vehicle_id, "trace": trace, "trace_df": trace_df})

    # 7.
    map = NxMap(parse_osmnx_graph(road_graph, network_type=NetworkType.DRIVE))
    for trajectory in trajectories:
        vehicle_id = trajectory["vehicle_id"]

        trace_df = trajectory["trace_df"]
        trace = Trace.from_dataframe(trace_df, lon_column="longitude", lat_column="latitude")

        matcher = LCSSMatcher(map)
        match_result = matcher.match_trace(trace)
        path_df = match_result.path_to_dataframe()

        pd.set_option('display.max_rows', None)

        # 8.
        path_nodes, path_edges = get_path(path_df)
        valid = is_path_valid(path_edges, road_graph)
        if valid != 0:
            print(f"Invalid matching path for vehicle {vehicle_id}")
            print("Index: ", valid)
            print(path_df)

        trajectory["path_nodes"] = path_nodes
        trajectory["path_edges"] = path_edges

    # 9.
    for trajectory in trajectories:
        vehicle_id = trajectory["vehicle_id"]
        path_nodes = trajectory["path_nodes"]

        fig, ax = ox.plot_graph_route(
            road_graph,
            path_nodes,
            figsize=(10, 10),
            node_size=10,
            show=False,
            save=True,
            filepath=f"{DATA_PATH}/trajectory/trajectory-{vehicle_id}.png"
        )
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pos = {node: (road_graph_og.nodes[node]["x"], road_graph_og.nodes[node]["y"]) for node in road_graph_og.nodes()}
        nx.draw_networkx_edges(road_graph_og, pos, arrows=False, ax=ax)

        trace = trajectory["trace"]
        trace = np.array(trace)
        ax.scatter(trace[:, 0], trace[:, 1], s=10, c="blue")

        fig.savefig(f"{DATA_PATH}/trajectory/records-{vehicle_id}.png")
        plt.close(fig)
    #
    # for trajectory in trajectories:
    #     del trajectory["trace"]
    #     del trajectory["trace_df"]
    #
    # # 10.
    # util.save(trajectories, f"{DATA_PATH}/trajectory/record_trajectories.pickle")

# 19, 29, 43, 50, 52, ...
