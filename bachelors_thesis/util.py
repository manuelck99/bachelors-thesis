import base64

import Levenshtein as ls
import numpy as np


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

    edit_distance = ls.distance(s1, s2)
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
