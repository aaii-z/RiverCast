import numpy as np
import pandas as pd

from rivercast.network import RiverNetwork
from rivercast.pathogens import Pathogen


def spill_flow(rain: np.ndarray, capacity_m3s: np.ndarray, cfg: dict) -> np.ndarray:
    """Storm-overflow discharge (m3/s), saturating once 6-hour rain exceeds the threshold."""
    rain_6h = pd.DataFrame(rain).rolling(6, min_periods=1).sum().to_numpy()
    excess = np.maximum(rain_6h - cfg["spill_rain_6h_mm"], 0.0)
    return capacity_m3s * (1 - np.exp(-excess / cfg["spill_scale_mm"]))


def build_loads(net: RiverNetwork, static: pd.DataFrame, flow: np.ndarray, rain: np.ndarray,
                times: pd.DatetimeIndex, pathogens: list[Pathogen], cfg: dict) -> np.ndarray:
    """Hourly loads [T, N, P] (organisms/h) from effluent, storm overflows, farm runoff,
    bed resuspension and the upstream boundary inflow at head reaches."""
    urban = static.urban_frac.to_numpy()
    wwtw = static.wwtw_m3s.to_numpy()
    reach = net.nodes.reach_m.to_numpy()

    cso_capacity = (cfg["cso_m3s_full_urban"] * urban * (urban >= cfg["cso_urban_threshold"])
                    + cfg["wwtw_storm_ratio"] * wwtw)
    spill_m3h = spill_flow(rain, cso_capacity, cfg) * 3600
    effluent_m3h = wwtw * 3600
    runoff_m3h = cfg["runoff_coeff"] * rain / 1000 * reach * cfg["runoff_width_m"] * (1 - urban)
    rise_m3h = np.maximum(np.diff(flow, axis=0, prepend=flow[:1]), 0.0) * 3600

    heads = np.zeros(net.size, dtype=bool)
    heads[net.heads()] = True
    boundary_m3h = flow * 3600 * heads
    flow_ratio = flow / np.median(flow, axis=0)
    boundary_shape = flow_ratio ** cfg["boundary_gamma"]

    doy = times.dayofyear.to_numpy()[:, None]
    loads = np.empty((*flow.shape, len(pathogens)))
    for k, p in enumerate(pathogens):
        season = p.seasonal(doy)
        concentration_x_volume = (
            p.effluent * season * effluent_m3h
            + p.sewage * season * spill_m3h
            + p.runoff * runoff_m3h
            + p.resuspension * rise_m3h
            + p.background * boundary_shape * boundary_m3h
        )
        loads[..., k] = p.unit_per_m3 * concentration_x_volume
    return loads
