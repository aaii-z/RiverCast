import json
from pathlib import Path

import numpy as np
import pandas as pd

from rivercast import features, sources
from rivercast.dataset import Dataset
from rivercast.fetch import ea, osm
from rivercast.fetch.http import cached_frame, cached_json
from rivercast.fetch.weather import hourly_weather
from rivercast.geo import to_wgs84
from rivercast.network import RiverNetwork, build_network
from rivercast.observe import observe
from rivercast.pathogens import load_pathogens
from rivercast.transport import route


def build_dataset(cfg: dict) -> Dataset:
    raw = Path(cfg["paths"]["raw"])
    bbox = cfg["bbox"]
    river_names = [r["name"] for r in cfg["network"]["rivers"]]

    ways = cached_json(raw / "osm_rivers.json", lambda: osm.river_ways(river_names, bbox))
    plants = cached_json(raw / "osm_wwtw.json", lambda: osm.wastewater_plants(bbox))
    urban_rings = cached_json(raw / "osm_urban.json", lambda: osm.urban_areas(bbox))
    net = build_network(ways, cfg["network"])
    print(f"network: {net.size} nodes, {len(net.edges())} edges")

    period = cfg["period"]
    start = pd.Timestamp(period["start"]) - pd.Timedelta(days=period["spinup_days"])
    end = pd.Timestamp(period["end"])
    times = pd.date_range(start, end, freq="h", inclusive="left")

    flow_measures = {g["river"]: g["measure"] for g in cfg["flow_gauges"]}
    rain_measures = {g["station"]: g["measure"] for g in cfg["rain_gauges"]}
    w = cfg["weather"]
    flow = cached_frame(raw / "ea_flow.csv",
                        lambda: ea.hourly_frame(flow_measures, start, end, "mean"))
    rain = cached_frame(raw / "ea_rain.csv",
                        lambda: ea.hourly_frame(rain_measures, start, end, "sum"))
    weather = cached_frame(raw / "weather.csv",
                           lambda: hourly_weather(w["lon"], w["lat"], start, end))
    _report_missing(flow=flow, rain=rain, weather=weather)

    catchment = cfg["catchment"]
    urban = features.urban_fraction(net, urban_rings, catchment["urban_radius_m"])
    effluent, plants_used = features.wwtw_effluent(net, plants, catchment)
    static = features.static_features(net, urban, effluent)

    node_flow = features.node_flow(net, features.fill_flow(flow))
    node_rain = features.node_rain(net, features.fill_rain(rain), cfg["rain_gauges"])
    weather = weather.interpolate().bfill().ffill()
    water_temp = features.water_temperature(weather.air_temp, w["water_temp_tau_h"]).to_numpy()
    radiation = weather.radiation.to_numpy()
    shape = node_flow.shape
    dynamic = np.stack([node_rain, node_flow, np.broadcast_to(water_temp[:, None], shape),
                        np.broadcast_to(radiation[:, None], shape)], axis=-1)

    pathogens = load_pathogens(cfg["pathogens"])
    loads = sources.build_loads(net, static, node_flow, node_rain, times, pathogens, cfg["sources"])
    decay = np.stack([p.decay_rate(water_temp, radiation)[:, None].repeat(net.size, 1)
                      for p in pathogens], axis=-1)
    conc = route(loads, node_flow, decay, net.nodes.reach_m.to_numpy(), net.down, cfg["hydraulics"])
    target = conc / np.array([p.unit_per_m3 for p in pathogens])

    rng = np.random.default_rng(cfg["observation"]["seed"])
    observed, monitored = observe(target, pathogens, cfg["observation"], rng)

    keep = times >= pd.Timestamp(period["start"])
    nodes = _node_table(net, static, monitored, cfg["flow_gauges"])
    meta = {
        "name": cfg["name"],
        "dynamic_features": list(features.DYNAMIC_FEATURES),
        "static_features": list(features.STATIC_FEATURES),
        "targets": [p.name for p in pathogens],
        "units": {p.name: p.unit for p in pathogens},
        "lod": {p.name: p.lod for p in pathogens},
        "split": cfg["split"],
        "wwtw": plants_used,
        "edges": net.edges().to_dict("list"),
    }
    ds = Dataset(times=times[keep], dynamic=dynamic[keep], static=static.to_numpy(),
                 target=target[keep], observed=observed[keep], adjacency=net.adjacency(),
                 nodes=nodes, meta=meta)
    out = Path(cfg["paths"]["processed"])
    ds.save(out)
    _write_geojson(net, out / "rivers.geojson")
    return ds


def _node_table(net: RiverNetwork, static: pd.DataFrame, monitored: np.ndarray,
                gauges: list[dict]) -> pd.DataFrame:
    gauge_at = {node: river for river, node in features.gauge_nodes(net, gauges).items()}
    table = net.nodes[["id", "river", "lon", "lat", "chainage_m", "reach_m"]].copy()
    table["down"] = net.down
    table["monitored"] = monitored
    table["flow_gauge"] = [i in gauge_at for i in range(net.size)]
    return pd.concat([table, static.reset_index(drop=True)], axis=1)


def _write_geojson(net: RiverNetwork, path: Path) -> None:
    feats = []
    for name, line in net.centrelines.items():
        x, y = np.array(line.coords).T
        lon, lat = to_wgs84(x, y)
        feats.append({"type": "Feature", "properties": {"name": name},
                      "geometry": {"type": "LineString",
                                   "coordinates": np.column_stack([lon, lat]).round(6).tolist()}})
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def _report_missing(**frames: pd.DataFrame) -> None:
    for name, df in frames.items():
        pct = (df.isna().mean() * 100).round(1)
        print(f"missing % {name}: " + ", ".join(f"{k}={v}" for k, v in pct.items()))
