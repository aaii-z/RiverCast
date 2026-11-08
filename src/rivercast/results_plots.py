import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from rivercast.plots import BLUES, GRID, INK, INK_2, MUTED

MODEL_STYLE = {
    "persistence": ("Persistence", "#2a78d6"),
    "xgboost": ("XGBoost", "#eb6834"),
    "lstm": ("LSTM", "#1baf7a"),
    "graph_wavenet": ("Graph WaveNet", "#6250d6"),
    "graph_wavenet_single": ("Graph WaveNet, single-task", "#e87ba4"),
}


def plot_mae_by_horizon(metrics: pd.DataFrame, nodes: str = "all"):
    targets = list(dict.fromkeys(metrics.target))
    fig, axes = plt.subplots(1, len(targets), figsize=(11, 3.6), sharey=False)
    for ax, target in zip(axes, targets, strict=True):
        sub = metrics[(metrics.target == target) & (metrics.nodes == nodes)]
        for model, (label, color) in MODEL_STYLE.items():
            rows = sub[sub.model == model].sort_values("horizon_h")
            if len(rows):
                ax.plot(rows.horizon_h, rows.mae, color=color, marker="o", ms=5, label=label)
        ax.set(title=target, xlabel="Forecast horizon (h)", xticks=[1, 3, 6, 12])
    axes[0].set_ylabel("MAE (log10)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.suptitle(f"Forecast error by horizon ({nodes} nodes)", color=INK, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_node_groups(metrics: pd.DataFrame):
    table = metrics.pivot_table(index="nodes", columns="model", values="mae")
    groups = ["monitored", "unmonitored"]
    models = [m for m in MODEL_STYLE if m in table.columns]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        label, color = MODEL_STYLE[model]
        x = np.arange(len(groups)) + i * width
        values = table.loc[groups, model].to_numpy()
        ax.bar(x, values, width * 0.92, color=color, label=label)
        for xi, v in zip(x, values, strict=True):
            ax.annotate(f"{v:.2f}", (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK_2)
    ax.set_xticks(np.arange(len(groups)) + 0.4 - width / 2,
                  ["Nodes with a sensor", "Nodes without a sensor"])
    ax.set(ylabel="MAE (log10)", title="Where does the river graph help?")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    return fig


def plot_forecast_series(times, truth, observed, forecasts: dict, node_id: str, unit: str,
                         horizon: int, rain=None):
    """forecasts: model -> (median, lo, hi) arrays aligned with `times` (target times)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                             gridspec_kw={"height_ratios": [1, 3]})
    if rain is not None:
        axes[0].bar(times, rain, width=1 / 24, color="#2a78d6")
    axes[0].set(ylabel="mm/h", title="Catchment-mean rainfall")
    ax = axes[1]
    for model, (median, lo, hi) in forecasts.items():
        label, color = MODEL_STYLE[model]
        if lo is not None:
            ax.fill_between(times, lo, hi, color=color, alpha=0.18, lw=0,
                            label=f"{label} 80% band (calibrated)")
        ax.plot(times, median, color=color, label=f"{label} median")
    ax.plot(times, truth, color=INK, lw=1.6, ls="--", label="Truth")
    if np.isfinite(observed).any():
        ax.scatter(times, observed, s=10, color=INK, zorder=4, label="Sensor readings")
    ax.set(yscale="log", ylabel=unit,
           title=f"{horizon} h-ahead forecast at {node_id}"
                 + ("" if np.isfinite(observed).any() else " (no sensor at this node)"))
    ax.legend(loc="upper right", fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_forecast_map(nodes: pd.DataFrame, rivers_geojson: Path, truth, forecast,
                      unit: str, when: str, horizon: int):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    vmin = max(min(truth.min(), forecast.min()), 1)
    norm = LogNorm(vmin=vmin, vmax=max(truth.max(), forecast.max()))
    rivers = json.loads(rivers_geojson.read_text())["features"]
    for ax, values, title in ((axes[0], truth, "Truth (simulated)"),
                              (axes[1], forecast, f"Graph WaveNet, {horizon} h ahead")):
        for feat in rivers:
            lon, lat = np.array(feat["geometry"]["coordinates"]).T
            ax.plot(lon, lat, color="#cde2fb", lw=2.5, zorder=1)
        dots = ax.scatter(nodes.lon, nodes.lat, c=values, cmap=BLUES, norm=norm, s=80, zorder=3,
                          edgecolors=np.where(nodes.monitored, INK, MUTED), linewidths=1.0)
        ax.set(title=title, xlabel="Longitude", aspect=1 / np.cos(np.radians(52.6)))
        ax.grid(color=GRID)
    axes[0].set_ylabel("Latitude")
    fig.colorbar(dots, ax=axes, shrink=0.8, label=unit)
    fig.suptitle(f"FIO across Norwich at {when}", color=INK, fontweight="bold")
    return fig
