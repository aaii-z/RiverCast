import math
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from rivercast.geo import to_bng, to_wgs84


@dataclass
class RiverNetwork:
    """Directed river graph: each node is a well-mixed reach; `down[i]` is -1 at the outlet."""

    nodes: pd.DataFrame
    down: np.ndarray
    edge_m: np.ndarray
    centrelines: dict[str, LineString]

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def xy(self) -> np.ndarray:
        return self.nodes[["x", "y"]].to_numpy()

    def edges(self) -> pd.DataFrame:
        src = np.flatnonzero(self.down >= 0)
        return pd.DataFrame({"src": src, "dst": self.down[src], "length_m": self.edge_m[src]})

    def adjacency(self) -> np.ndarray:
        a = np.zeros((self.size, self.size), dtype=np.float32)
        e = self.edges()
        a[e.src, e.dst] = 1.0
        return a

    def downstream_path(self, node: int) -> list[int]:
        path = [node]
        while self.down[path[-1]] >= 0:
            path.append(int(self.down[path[-1]]))
        return path

    def upstream_mask(self) -> np.ndarray:
        """mask[i, j] is True when i is j or drains into j."""
        mask = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.size):
            mask[i, self.downstream_path(i)] = True
        return mask

    def heads(self) -> np.ndarray:
        has_upstream = np.zeros(self.size, dtype=bool)
        has_upstream[self.down[self.down >= 0]] = True
        return np.flatnonzero(~has_upstream)


def centreline(lines: list[LineString], start: Point, end: Point,
               max_snap_m: float, max_gap_m: float = 30.0) -> LineString:
    """Shortest path through possibly braided, fragmented OSM ways from start to end."""
    g = nx.Graph()
    for line in lines:
        pts = [tuple(np.round(c, 1)) for c in line.coords]
        for a, b in zip(pts[:-1], pts[1:], strict=True):
            g.add_edge(a, b, w=math.dist(a, b))
    _bridge_gaps(g, max_gap_m)

    verts = np.array(g.nodes)
    ends = []
    for p in (start, end):
        d = np.hypot(verts[:, 0] - p.x, verts[:, 1] - p.y)
        if d.min() > max_snap_m:
            raise ValueError(f"Point ({p.x:.0f}, {p.y:.0f}) is {d.min():.0f} m from the river")
        ends.append(tuple(verts[d.argmin()]))
    return LineString(nx.shortest_path(g, *ends, weight="w"))


def _bridge_gaps(g: nx.Graph, max_gap_m: float) -> None:
    components = list(nx.connected_components(g))
    if len(components) == 1:
        return
    label = {v: i for i, comp in enumerate(components) for v in comp}
    verts = list(g.nodes)
    xy = np.array(verts)
    labels = np.array([label[v] for v in verts])
    for v in [v for v in verts if g.degree(v) == 1]:
        d = np.hypot(xy[:, 0] - v[0], xy[:, 1] - v[1])
        d[labels == label[v]] = np.inf
        k = int(d.argmin())
        if d[k] <= max_gap_m:
            g.add_edge(v, verts[k], w=float(d[k]))


def build_network(ways: list[dict], cfg: dict) -> RiverNetwork:
    lines: dict[str, list[LineString]] = {}
    for way in ways:
        lon, lat = np.array(way["coords"]).T
        lines.setdefault(way["name"], []).append(LineString(np.column_stack(to_bng(lon, lat))))

    spacing = cfg["spacing_km"] * 1000
    rows, down, edge_m, centrelines = [], [], [], {}
    for river in cfg["rivers"]:
        name = river["name"]
        start, end = (Point(to_bng(*river[k])) for k in ("start", "end"))
        line = centreline(lines[name], start, end, cfg["max_snap_m"])
        centrelines[name] = line

        n = max(1, round(line.length / spacing))
        reach = line.length / n
        first = len(rows)
        code = name.split()[-1][:3].upper()
        for i in range(n):
            p = line.interpolate((i + 0.5) * reach)
            rows.append({"id": f"{code}{i + 1:02d}", "river": name, "x": p.x, "y": p.y,
                         "chainage_m": (i + 0.5) * reach, "reach_m": reach})
            down.append(first + i + 1)
            edge_m.append(reach)

        if "joins" in river:
            down[-1], edge_m[-1] = _confluence(rows, centrelines[river["joins"]],
                                               river["joins"], Point(line.coords[-1]), reach)
        else:
            down[-1], edge_m[-1] = -1, np.nan

    nodes = pd.DataFrame(rows)
    nodes["lon"], nodes["lat"] = to_wgs84(nodes.x.to_numpy(), nodes.y.to_numpy())
    return RiverNetwork(nodes, np.array(down), np.array(edge_m), centrelines)


def _confluence(rows: list[dict], parent_line: LineString, parent: str,
                mouth: Point, reach: float) -> tuple[int, float]:
    """Parent node whose reach contains the confluence, and the edge length to it."""
    along = parent_line.project(mouth)
    candidates = [(abs(r["chainage_m"] - along), i) for i, r in enumerate(rows)
                  if r["river"] == parent]
    offset, j = min(candidates)
    return j, reach / 2 + offset
