import numpy as np
import pandas as pd
from shapely import make_valid, unary_union
from shapely.geometry import Point, Polygon

from rivercast.geo import nearest, to_bng
from rivercast.network import RiverNetwork

STATIC_FEATURES = ("dist_to_outlet_km", "upstream_km", "urban_frac", "wwtw_m3s")
DYNAMIC_FEATURES = ("rain_mm", "flow_m3s", "water_temp_c", "radiation_wm2")


def _project(points: list[dict]) -> np.ndarray:
    lon = np.array([p["lon"] for p in points])
    lat = np.array([p["lat"] for p in points])
    return np.column_stack(to_bng(lon, lat))


def urban_fraction(net: RiverNetwork, rings: list, radius_m: float) -> np.ndarray:
    polygons = []
    for ring in rings:
        lon, lat = np.array(ring).T
        polygons.append(make_valid(Polygon(np.column_stack(to_bng(lon, lat)))))
    urban = unary_union(polygons)
    fractions = []
    for x, y in net.xy:
        area = Point(x, y).buffer(radius_m)
        fractions.append(area.intersection(urban).area / area.area)
    return np.array(fractions)


def wwtw_effluent(net: RiverNetwork, plants: list[dict], cfg: dict) -> tuple[np.ndarray, list]:
    """Dry-weather effluent (m3/s) discharged at each node by nearby treatment works."""
    idx, dist = nearest(_project(plants), net.xy)
    effluent = np.zeros(net.size)
    assigned = []
    for plant, node, d in zip(plants, idx, dist, strict=True):
        if d > cfg["wwtw_max_dist_m"]:
            continue
        q = next((v for k, v in cfg["wwtw_m3s"].items() if k in plant["name"]),
                 cfg["wwtw_default_m3s"])
        effluent[node] += q
        assigned.append({"name": plant["name"] or "unnamed", "node": int(node), "m3s": q})
    return effluent, assigned


def static_features(net: RiverNetwork, urban: np.ndarray, effluent: np.ndarray) -> pd.DataFrame:
    edge = np.nan_to_num(net.edge_m)
    reach = net.nodes.reach_m.to_numpy()
    upstream = net.upstream_mask()
    return pd.DataFrame({
        "dist_to_outlet_km": [edge[net.downstream_path(i)].sum() / 1000 for i in range(net.size)],
        "upstream_km": reach @ upstream / 1000,
        "urban_frac": urban,
        "wwtw_m3s": effluent,
    }, index=net.nodes.id)


def gauge_nodes(net: RiverNetwork, gauges: list[dict]) -> dict[str, int]:
    """Nearest node on the gauge's own river."""
    out = {}
    for g in gauges:
        on_river = np.flatnonzero(net.nodes.river.to_numpy() == g["river"])
        idx, _ = nearest(_project([g]), net.xy[on_river])
        out[g["river"]] = int(on_river[idx[0]])
    return out


def node_flow(net: RiverNetwork, flow: pd.DataFrame) -> np.ndarray:
    """Each river carries its gauged flow; flows add at confluences."""
    rivers = net.nodes.river.to_numpy()
    upstream = net.upstream_mask()
    q = np.zeros((len(flow), net.size))
    for j in range(net.size):
        for river in set(rivers[upstream[:, j]]):
            q[:, j] += flow[river].to_numpy()
    return q


def node_rain(net: RiverNetwork, rain: pd.DataFrame, gauges: list[dict]) -> np.ndarray:
    idx, _ = nearest(net.xy, _project(gauges))
    names = [g["station"] for g in gauges]
    return rain[names].to_numpy()[:, idx]


def water_temperature(air: pd.Series, tau_h: float) -> pd.Series:
    return air.ewm(alpha=1 / tau_h).mean().clip(lower=1.0)


def fill_flow(flow: pd.DataFrame, max_gap_h: int = 24) -> pd.DataFrame:
    return flow.interpolate(limit=max_gap_h, limit_area="inside").interpolate().bfill().ffill()


def fill_rain(rain: pd.DataFrame) -> pd.DataFrame:
    """Gaps take the mean of the other gauges at that hour, else zero."""
    return rain.apply(lambda col: col.fillna(rain.mean(axis=1))).fillna(0.0)
