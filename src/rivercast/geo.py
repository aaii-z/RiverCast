import numpy as np
from pyproj import Transformer

_WGS84_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def to_bng(lon, lat):
    """WGS84 lon/lat -> British National Grid metres."""
    return _WGS84_TO_BNG.transform(lon, lat)


def to_wgs84(x, y):
    return _WGS84_TO_BNG.transform(x, y, direction="INVERSE")


def nearest(points: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of and distance to the nearest target for each point (both [n, 2])."""
    d = np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    return idx, d[np.arange(len(points)), idx]
