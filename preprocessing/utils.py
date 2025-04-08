import numpy as np
from pyproj import Transformer

opt_params = [-4.25994981e-06, -8.74119271e-06, 8.11700876e-06, -3.95042166e-06, 1.14014210e+02, 2.26438070e+01]

EPSG_4326 = "EPSG:4326"
EPSG_32650 = "EPSG:32650"


def xy_to_epsg4326(xy_points: np.ndarray, *, params=opt_params) -> np.ndarray:
    a, b, c, d, e, f = params
    x, y = xy_points[:, 0], xy_points[:, 1]
    lon = a * x + b * y + e
    lat = c * x + d * y + f
    return np.vstack([lon, lat]).T


def epsg4326_to_epsg32650(gps_points: np.ndarray) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
    lon, lat = gps_points[:, 0], gps_points[:, 1]
    lon_t, lat_t = transformer.transform(lon, lat)
    return np.vstack([lon_t, lat_t]).T


def epsg32650_to_epsg4326(gps_points: np.ndarray) -> np.ndarray:
    transformer = Transformer.from_crs("EPSG:32650", "EPSG:4326", always_xy=True)
    lon, lat = gps_points[:, 0], gps_points[:, 1]
    lon_t, lat_t = transformer.transform(lon, lat)
    return np.vstack([lon_t, lat_t]).T
