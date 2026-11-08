import numpy as np
import pytest

from rivercast.geo import to_wgs84
from rivercast.network import build_network

# Synthetic Y-shaped system in British National Grid metres: a tributary joins a main river.
ORIGIN = (620000.0, 305000.0)


def _way(name, pts):
    x, y = np.array(pts).T + np.array(ORIGIN)[:, None]
    lon, lat = to_wgs84(x, y)
    return {"name": name, "coords": list(zip(lon, lat, strict=True))}


def _lonlat(pt):
    return list(to_wgs84(ORIGIN[0] + pt[0], ORIGIN[1] + pt[1]))


@pytest.fixture
def net():
    ways = [
        _way("Main", [(0, 0), (5000, 0)]),
        _way("Main", [(5000, 0), (10000, 0)]),  # split way, shares a vertex
        _way("Trib", [(5000, 4000), (5000, 0)]),
    ]
    cfg = {"spacing_km": 2.5, "max_snap_m": 100, "rivers": [
        {"name": "Main", "start": _lonlat((0, 0)), "end": _lonlat((10000, 0))},
        {"name": "Trib", "start": _lonlat((5000, 4000)), "end": _lonlat((5000, 0)),
         "joins": "Main"},
    ]}
    return build_network(ways, cfg)


def test_topology(net):
    assert net.size == 4 + 2
    assert list(net.down[:4]) == [1, 2, 3, -1]
    trib_mouth = int(np.flatnonzero(net.nodes.river == "Trib")[-1])
    assert net.down[trib_mouth] in (1, 2)
    assert list(net.heads()) == [0, 4]


def test_every_node_reaches_outlet(net):
    assert all(net.downstream_path(i)[-1] == 3 for i in range(net.size))
    assert net.upstream_mask()[:, 3].all()


def test_far_endpoint_rejected():
    ways = [_way("Main", [(0, 0), (1000, 0)])]
    cfg = {"spacing_km": 1, "max_snap_m": 100, "rivers": [
        {"name": "Main", "start": _lonlat((0, 0)), "end": _lonlat((1000, 5000))}]}
    with pytest.raises(ValueError):
        build_network(ways, cfg)
