import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rivercast import analysis
from rivercast.config import load_config
from rivercast.evaluate import forecast_metrics
from rivercast.experiment import load_setup, predict_single_task, predict_split, run_dir
from rivercast.models import MODELS
from rivercast.results_plots import (
    MODEL_STYLE,
    plot_forecast_map,
    plot_forecast_series,
    plot_mae_by_horizon,
    plot_node_groups,
)

FIO_ALERT = 1000.0  # CFU/100 mL, the threshold used by the UK Open E. coli models
SINGLE = "graph_wavenet_single"


def load_all(cfg, data, split):
    preds = {m: predict_split(cfg, data, m, split) for m in MODELS
             if (run_dir(cfg, m) / "predictions.npz").exists()}
    targets = data.ds.meta["targets"]
    if all((run_dir(cfg, "graph_wavenet", target=t) / "predictions.npz").exists()
           for t in targets):
        preds[SINGLE] = predict_single_task(cfg, data, "graph_wavenet", split)
    return preds


def storm_window(data, start, hours=96):
    rain = pd.Series(data.ds.dynamic[..., 0].mean(axis=1), index=data.ds.times)
    test = rain[data.ds.split("test")]
    peak = test.rolling(24).sum().idxmax() - pd.Timedelta(hours=24)
    first = max(peak - pd.Timedelta(hours=12), data.ds.times[start])
    return data.ds.times.get_indexer(pd.date_range(first, periods=hours, freq="h"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: calibration, storms, alerts, figures.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--horizon", type=int, default=6, help="horizon (h) for alerts/figures")
    parser.add_argument("--node", default="WEN06", help="node for the forecast figure")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--export", default="docs/forecast_data.json")
    args = parser.parse_args()

    cfg, data = load_setup(args.config)
    ds, targets = data.ds, data.ds.meta["targets"]
    out, figures = Path(cfg["results_dir"]), Path(args.figures)
    out.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    test = load_all(cfg, data, "test")
    origins = test["graph_wavenet"][0]
    truth = data.targets(origins, truth=True)
    observed = data.observed_log10(origins)
    j = list(data.horizons).index(args.horizon)
    print("models:", ", ".join(test))

    # Calibration: conformal widening fitted on validation observations.
    calibrated, cov_rows = {}, []
    for model, (_, pred, quantiles) in test.items():
        if len(quantiles) == 1:
            continue
        val_origins, val_pred, _ = (predict_single_task(cfg, data, "graph_wavenet", "val")
                                    if model == SINGLE else predict_split(cfg, data, model, "val"))
        offsets = analysis.conformal_offsets(val_pred, data.observed_log10(val_origins),
                                             quantiles[-1] - quantiles[0])
        calibrated[model] = analysis.widen(pred, offsets)
        for stage, p in (("raw", pred), ("calibrated", calibrated[model])):
            for reference, ref in (("truth", truth), ("observations", observed)):
                table = analysis.coverage(p, ref).assign(target=targets)
                cov_rows.append(table.assign(model=model, stage=stage, reference=reference))
    cov = pd.concat(cov_rows, ignore_index=True)

    # Accuracy, overall and by regime.
    metrics = pd.concat([forecast_metrics(p, q, truth, data.monitored, targets, data.horizons)
                         .assign(model=m) for m, (_, p, q) in test.items()], ignore_index=True)
    train_truth = data.truth_log10[ds.split("train")]
    high = analysis.high_pollution_mask(truth, train_truth)
    regimes = pd.concat([analysis.mae_by_regime(p[..., q.index(0.5)], truth, high, targets)
                         .assign(model=m) for m, (_, p, q) in test.items()], ignore_index=True)

    # FIO exceedance alerts at the chosen horizon.
    k_fio, threshold = targets.index("fio"), np.log10(FIO_ALERT)
    alerts = []
    for model, (_, p, q) in test.items():
        median = p[:, :, j, k_fio, q.index(0.5)]
        alerts.append({"model": model, "rule": "median > 1000",
                       **analysis.alert_skill(median, truth[:, :, j, k_fio], threshold)})
        if model in calibrated:
            upper = calibrated[model][:, :, j, k_fio, -1]
            alerts.append({"model": model, "rule": "upper band > 1000",
                           **analysis.alert_skill(upper, truth[:, :, j, k_fio], threshold)})
    alerts = pd.DataFrame(alerts)

    write_report(out / "phase3.md", metrics, cov, regimes, alerts, args.horizon)
    metrics.to_csv(out / "phase3_metrics.csv", index=False)
    cov.to_csv(out / "phase3_coverage.csv", index=False)

    # Figures and the interactive-map export.
    plot_mae_by_horizon(metrics).savefig(figures / "mae_by_horizon.png", dpi=160,
                                         bbox_inches="tight")
    plot_node_groups(metrics[metrics.nodes != "all"]).savefig(
        figures / "node_groups.png", dpi=160, bbox_inches="tight")

    window = storm_window(data, origins[0] + args.horizon)
    s = window - args.horizon - origins[0]
    node = int(ds.nodes.index[ds.nodes.id == args.node][0])
    unit = ds.meta["units"]["fio"]
    gwn = calibrated["graph_wavenet"][s, :, j, k_fio]
    lstm_median = test["lstm"][1][s, :, j, k_fio, 1]
    truth_w = data.truth_log10[window, :, k_fio]
    rain = ds.dynamic[window][..., 0].mean(axis=1)
    times = ds.times[window]
    observed_w = data.observed_log10(window - args.horizon)[:, node, j, k_fio]
    plot_forecast_series(
        times, 10 ** truth_w[:, node], 10 ** observed_w,
        {"lstm": (10 ** lstm_median[:, node], None, None),
         "graph_wavenet": tuple(10 ** gwn[:, node, i] for i in (1, 0, 2))},
        args.node, unit, args.horizon, rain,
    ).savefig(figures / "forecast_event.png", dpi=160, bbox_inches="tight")

    peak = int(np.argmax(truth_w.mean(axis=1)))
    data_cfg = load_config(cfg["data_config"])
    plot_forecast_map(ds.nodes, Path(data_cfg["paths"]["processed"]) / "rivers.geojson",
                      10 ** truth_w[peak], 10 ** gwn[peak, :, 1], unit,
                      times[peak].strftime("%d %b %Y %H:00"), args.horizon,
                      ).savefig(figures / "forecast_map.png", dpi=160, bbox_inches="tight")

    export_map(Path(args.export), data_cfg, ds, times, rain, truth_w, gwn, lstm_median,
               unit, args.horizon)
    print((out / "phase3.md").read_text())


def write_report(path, metrics, cov, regimes, alerts, horizon):
    order = [m for m in MODEL_STYLE if m in set(metrics.model)]
    groups = metrics.pivot_table(index="nodes", columns="model", values="mae")[order]
    multi = (metrics[(metrics.nodes == "all") & metrics.model.isin(["graph_wavenet", SINGLE])]
             .pivot_table(index="target", columns="model", values="mae"))
    coverage = (cov[cov.reference == "truth"]
                .pivot_table(index="model", columns="stage", values="coverage")
                .reindex([m for m in order if m in set(cov.model)])[["raw", "calibrated"]])
    coverage_obs = (cov[cov.reference == "observations"]
                    .pivot_table(index="model", columns="stage", values="coverage")
                    .reindex(coverage.index)[["raw", "calibrated"]])
    width = (cov[cov.reference == "truth"]
             .pivot_table(index="model", columns="stage", values="width")
             .reindex(coverage.index)[["raw", "calibrated"]])
    per_target = (cov[cov.stage == "raw"]
                  .pivot_table(index=["model", "target"], columns="reference", values="coverage"))
    regime = regimes.pivot_table(index="model", values=["calm", "high_pollution"]).reindex(order)
    sections = [
        "# Phase 3 results (test: Jul-Dec 2024)",
        "## MAE (log10) by node group\n\n" + groups.round(3).to_markdown(),
        "## Multi-task vs single-task Graph WaveNet (MAE log10, all nodes)\n\n"
        + (multi.round(3).to_markdown() if SINGLE in multi.columns
           else "_single-task runs missing_"),
        "## 80% interval coverage vs truth\n\n" + coverage.round(3).to_markdown(),
        "## 80% interval coverage vs sensor observations\n\n" + coverage_obs.round(3).to_markdown(),
        "## Mean interval width (log10)\n\n" + width.round(3).to_markdown(),
        "## Raw 80% coverage per target\n\nNorovirus is mostly below the detection limit, so "
        "sensors report LOD/2 and models learn that value: well covered against observations, "
        "biased high against the (lower) truth.\n\n" + per_target.round(3).to_markdown(),
        "## MAE (log10) in calm vs high-pollution hours (truth above node's 95th percentile)\n\n"
        + regime.round(3).to_markdown(),
        f"## FIO > 1000 CFU/100 mL alerts, {horizon} h ahead\n\n"
        + alerts.round(3).to_markdown(index=False),
    ]
    path.write_text("\n\n".join(sections) + "\n")


def export_map(path, data_cfg, ds, times, rain, truth, gwn, lstm_median, unit, horizon):
    rivers = json.loads((Path(data_cfg["paths"]["processed"]) / "rivers.geojson").read_text())

    def sig(a):
        return np.round(10 ** np.asarray(a, dtype=float), 1).tolist()

    payload = {
        "unit": unit, "horizon_h": horizon, "threshold": FIO_ALERT,
        "times": [t.strftime("%Y-%m-%dT%H:00") for t in times],
        "rain": np.round(rain, 2).tolist(),
        "nodes": ds.nodes[["id", "river", "lon", "lat", "monitored", "urban_frac", "down"]]
        .round(5).to_dict("records"),
        "rivers": [f["geometry"]["coordinates"][::3] + [f["geometry"]["coordinates"][-1]]
                   for f in rivers["features"]],
        "truth": sig(truth), "gwn_median": sig(gwn[..., 1]), "gwn_lo": sig(gwn[..., 0]),
        "gwn_hi": sig(gwn[..., 2]), "lstm_median": sig(lstm_median),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
