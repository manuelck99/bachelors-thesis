import base64
import pickle
from typing import Any

import networkx as nx
import numpy as np
from Levenshtein import distance
from pyproj import Transformer

OPT_PARAMS: list[float] = [-4.25994981e-06, -8.74119271e-06, 8.11700876e-06, -3.95042166e-06, 1.14014210e+02,
                           2.26438070e+01]

EPSG_4326: str = "EPSG:4326"
EPSG_32650: str = "EPSG:32650"


def xy_to_epsg4326(xy_points: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = OPT_PARAMS
    x, y = xy_points[:, 0], xy_points[:, 1]
    lon = a * x + b * y + e
    lat = c * x + d * y + f
    return np.vstack([lon, lat]).T


def epsg4326_to_epsg32650(gps_points: np.ndarray) -> np.ndarray:
    transformer = Transformer.from_crs(EPSG_4326, EPSG_32650, always_xy=True)
    lon, lat = gps_points[:, 0], gps_points[:, 1]
    lon_t, lat_t = transformer.transform(lon, lat)
    return np.vstack([lon_t, lat_t]).T


def epsg32650_to_epsg4326(gps_points: np.ndarray) -> np.ndarray:
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
    else:
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


def clip(v: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
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
