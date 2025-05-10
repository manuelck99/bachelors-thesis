from __future__ import annotations

import base64
import pickle
from typing import Any, TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from Levenshtein import distance
from mappymatch.constructs.trace import Trace
from pyproj import Transformer

if TYPE_CHECKING:
    from vehicle_record import Record

OPT_PARAMS: list[float] = [-4.25994981e-06, -8.74119271e-06, 8.11700876e-06, -3.95042166e-06, 1.14014210e+02,
                           2.26438070e+01]

EPSG_4326: str = "EPSG:4326"
EPSG_32650: str = "EPSG:32650"


def xy_to_epsg4326(xy_points: np.ndarray) -> np.ndarray:
    """
    Transforms *xy_points* from unknown projection back into EPSG:4326 coordinates.

    :param xy_points: NumPy ``ndarray`` of points in unknown projection
    :return: transformed points in EPSG:4326 coordinates
    """

    a, b, c, d, e, f = OPT_PARAMS
    x, y = xy_points[:, 0], xy_points[:, 1]
    lon = a * x + b * y + e
    lat = c * x + d * y + f
    return np.vstack([lon, lat]).T


def epsg4326_to_epsg32650(gps_points: np.ndarray) -> np.ndarray:
    """
    Transforms *gps_points* from EPSG:4326 into EPSG:32650 coordinates.

    :param gps_points: NumPy ``ndarray`` of points in EPSG:4326 coordinates
    :return: transformed points in EPSG:32650 coordinates
    """

    transformer = Transformer.from_crs(EPSG_4326, EPSG_32650, always_xy=True)
    lon, lat = gps_points[:, 0], gps_points[:, 1]
    lon_t, lat_t = transformer.transform(lon, lat)
    return np.vstack([lon_t, lat_t]).T


def epsg32650_to_epsg4326(gps_points: np.ndarray) -> np.ndarray:
    """
        Transforms *gps_points* from EPSG:32650 into EPSG:4326 coordinates.

        :param gps_points: NumPy ``ndarray`` of points in EPSG:32650 coordinates
        :return: transformed points in EPSG:4326 coordinates
        """

    transformer = Transformer.from_crs(EPSG_32650, EPSG_4326, always_xy=True)
    lon, lat = gps_points[:, 0], gps_points[:, 1]
    lon_t, lat_t = transformer.transform(lon, lat)
    return np.vstack([lon_t, lat_t]).T


def save_graph(graph: nx.MultiDiGraph, path: str) -> None:
    save(graph, path)


def load_graph(path: str) -> nx.MultiDiGraph:
    return load(path)


def save(object: Any, path: str) -> None:
    with open(path, mode="wb") as file:
        pickle.dump(object, file)


def load(path: str) -> Any:
    with open(path, mode="rb") as file:
        return pickle.load(file)


def feature_from_base64(f: str | None = None) -> np.ndarray | None:
    """
    Turns a base64 encoded string into a NumPy ``ndarray``. If *f*
    is ``None``, ``None`` is returned. Additionally, if the generated ``ndarray``
    is all zeros, ``None`` is also returned.

    :param f: base64 encoded string, default: ``None``
    :return: NumPy ``ndarray`` or ``None``
    """

    if f is None:
        return None

    f = np.frombuffer(base64.b64decode(f), dtype=np.float32)
    if not np.any(f):
        # Feature is all zeros
        return None
    else:
        return f


def calculate_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """
    Returns the cosine similarity of *f1* and *f2*.

    :param f1: NumPy ``ndarray``
    :param f2: NumPy ``ndarray``
    :return: cosine similarity
    """

    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))


def normalize(f: np.ndarray) -> np.ndarray:
    """
    Returns L2 normalized *f*.

    :param f: NumPy ``ndarray``
    :return: L2 normalized *f*
    """

    norm = np.linalg.norm(f)
    return f / norm


def clip(v: float, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """
    Clips *v* into the range [*minimum*, *maximum*].

    :param v: value to clip
    :param minimum: lower bound of range, default: ``0.0``
    :param maximum: upper bound of range, default: ``1.0``
    :return: clipped value
    """

    return max(minimum, min(maximum, v))


def edit_distance_gain(s1: str, s2: str) -> float:
    """
    Returns the edit distance gain between *s1* and *s2*. The edit distance
    gain depends on the Levenshtein edit distance between two strings and
    is defined manually.

    :param s1: first string
    :param s2: second string
    :return: edit distance gain
    """

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


def get_trace(records: list[Record], road_graph: nx.MultiDiGraph, cameras_info: dict, *, project=True) -> Trace:
    trace = list()
    for record in records:
        x, y = record.get_coordinates(road_graph, cameras_info)
        trace.append([x, y])

    trace_df = pd.DataFrame(trace, columns=["longitude", "latitude"])
    return Trace.from_dataframe(trace_df, lon_column="longitude", lat_column="latitude", xy=project)


def get_path(road_graph: nx.MultiDiGraph, path_df: pd.DataFrame) -> list[tuple[int, int, int]] | None:
    """
    Returns a path as an edge ``list``. This function doesn't check if *path_df* is empty. Additionally, if
    one of the edges in *path_df* doesn't exist in *road_graph*, ``None`` is returned. Likewise, if the resulting
    edge ``list`` is not continuous.

    :param road_graph: NetworkX ``M̀ultiDiGraph``
    :param path_df: Pandas ``DataFrame`` returned from mappymatch's ``MatchResult.path_to_dataframe()``
    :return: path as an edge ``list``
    """

    edges = list()
    for _, (o, d, k) in path_df[["origin_junction_id", "destination_junction_id", "road_key"]].iterrows():
        edges.append((o, d, k))
        if not road_graph.has_edge(o, d, k):
            return None

    if all(map(lambda e1, e2: e1[1] == e2[0], edges, edges[1:])):
        return edges
    else:
        return None


def get_node_path(path: list[tuple[int, int, int]]) -> list[int]:
    """
    Returns a path as a node ``list``. This function doesn't check the validity of *path*.

    :param path: path as an edge ``list``
    :return: path as a node ``list``
    """

    return list(map(lambda e: e[0], path)) + list(map(lambda e: e[1], path[-1:]))
