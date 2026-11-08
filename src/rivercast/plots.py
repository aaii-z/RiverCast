import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from rivercast.dataset import Dataset

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2,
})


def downstream_path(nodes: pd.DataFrame, start_id: str) -> list[int]:
    path = [int(nodes.index[nodes.id == start_id][0])]
    while nodes.down[path[-1]] >= 0:
        path.append(int(nodes.down[path[-1]]))
    return path


def largest_rain_event(ds: Dataset, hours: int = 24) -> pd.Timestamp:
    rain = pd.Series(ds.dynamic[..., 0].mean(axis=1), index=ds.times)
    return rain.rolling(hours).sum().idxmax() - pd.Timedelta(hours=hours)


def plot_network(ds: Dataset, rivers_geojson: Path, ax=None):
    ax = ax or plt.subplots(figsize=(8, 5.5))[1]
    for feat in json.loads(rivers_geojson.read_text())["features"]:
        lon, lat = np.array(feat["geometry"]["coordinates"]).T
        ax.plot(lon, lat, color="#9ec5f4", lw=2.5, zorder=1, solid_capstyle="round")

    n = ds.nodes
    for _, row in n[n.down >= 0].iterrows():
        dst = n.loc[row.down]
        ax.annotate("", xy=(dst.lon, dst.lat), xytext=(row.lon, row.lat), zorder=2,
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.8,
                                "shrinkA": 5, "shrinkB": 5})
    dots = ax.scatter(n.lon, n.lat, c=n.urban_frac, cmap=BLUES, vmin=0, vmax=0.7, s=70,
                      edgecolors=np.where(n.monitored, INK, "#fcfcfb"), linewidths=1.2, zorder=3)
    wwtw = n[n.wwtw_m3s > 0]
    ax.scatter(wwtw.lon, wwtw.lat + 0.006, marker="v", s=40 + 150 * wwtw.wwtw_m3s, color=INK,
               zorder=4, label="Sewage works (OSM)")
    gauges = n[n.flow_gauge]
    ax.scatter(gauges.lon, gauges.lat - 0.006, marker="s", s=28, color=INK_2, zorder=4,
               label="EA flow gauge")
    ax.scatter([], [], s=70, facecolors="none", edgecolors=INK, linewidths=1.2,
               label="Monitored node (outlined)")
    for _, row in n.iterrows():
        ax.annotate(row.id, (row.lon, row.lat), xytext=(5, 4), textcoords="offset points",
                    fontsize=6.5, color=INK_2)

    ax.figure.colorbar(dots, ax=ax, shrink=0.7, label="Urban fraction (1 km)")
    ax.set(title="RiverCast Norwich network: Yare, Wensum, Tud, Tas",
           xlabel="Longitude", ylabel="Latitude", aspect=1 / np.cos(np.radians(52.6)))
    ax.legend(loc="lower right")
    return ax


def plot_event(ds: Dataset, start: pd.Timestamp, path_start: str = "WEN01",
               hours: int = 96, target: str = "fio"):
    window = (ds.times >= start) & (ds.times < start + pd.Timedelta(hours=hours))
    times = ds.times[window]
    path = downstream_path(ds.nodes, path_start)
    k = ds.meta["targets"].index(target)
    unit = ds.meta["units"][target]
    conc = ds.target[window][:, path, k]
    to_outlet = ds.nodes.dist_to_outlet_km.to_numpy()[path]
    dist_km = to_outlet[0] - to_outlet

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6, 2.2]})
    rain = ds.dynamic[window][..., 0].mean(axis=1)
    axes[0].bar(times, rain, width=1 / 24, color=SERIES[0])
    axes[0].set(title="Catchment-mean rainfall", ylabel="mm/h")

    picks = [0, len(path) // 2, len(path) - 1]
    for color, i in zip(SERIES, picks, strict=True):
        node_id = ds.nodes.id[path[i]]
        axes[1].plot(times, conc[:, i], color=color, label=f"{node_id} ({dist_km[i]:.0f} km)")
    axes[1].set(yscale="log", ylabel=unit,
                title=f"{target.upper()} at three points along the Wensum -> Yare path")
    axes[1].legend(loc="upper right")

    mesh = axes[2].pcolormesh(times, dist_km, conc.T, cmap=BLUES, shading="nearest",
                              norm=LogNorm(vmin=max(conc.min(), 1), vmax=conc.max()))
    axes[2].set(ylabel="Distance downstream (km)", title="Pollution wave travelling downstream")
    axes[2].invert_yaxis()
    axes[2].grid(False)
    fig.colorbar(mesh, cax=axes[2].inset_axes([1.01, 0, 0.015, 1]), label=unit)
    fig.autofmt_xdate()
    return fig


def plot_observations(ds: Dataset, node_id: str, start: pd.Timestamp, days: int = 21):
    window = (ds.times >= start) & (ds.times < start + pd.Timedelta(days=days))
    i = int(ds.nodes.index[ds.nodes.id == node_id][0])
    targets = ds.meta["targets"]
    fig, axes = plt.subplots(len(targets), 1, figsize=(9, 7), sharex=True)
    for ax, color, (k, name) in zip(axes, SERIES, enumerate(targets), strict=True):
        ax.plot(ds.times[window], ds.target[window, i, k], color=color, label="Simulated truth")
        ax.scatter(ds.times[window], ds.observed[window, i, k], s=10, color=INK, zorder=3,
                   label="Observed (noisy, sparse)")
        ax.axhline(ds.meta["lod"][name], color=MUTED, lw=1, ls="--", label="Detection limit")
        ax.set(yscale="log", ylabel=ds.meta["units"][name], title=f"{name} at {node_id}")
    axes[0].legend(loc="lower left", ncols=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
