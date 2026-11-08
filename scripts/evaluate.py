import argparse
from pathlib import Path

import pandas as pd

from rivercast.evaluate import forecast_metrics
from rivercast.experiment import load_predictions, load_setup, run_dir
from rivercast.models import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Score saved test predictions.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--fast", action="store_true", help="score the --fast runs")
    args = parser.parse_args()

    cfg, data = load_setup(args.config)
    tables = []
    for name in MODELS:
        path = run_dir(cfg, name, args.fast)
        if not (path / "predictions.npz").exists():
            print(f"skipping {name}: no predictions in {path}")
            continue
        origins, pred, quantiles = load_predictions(path)
        truth = data.targets(origins, truth=True)
        metrics = forecast_metrics(pred, quantiles, truth, data.monitored,
                                   data.ds.meta["targets"], data.horizons)
        tables.append(metrics.assign(model=name))

    results = pd.concat(tables, ignore_index=True)
    out = Path(cfg["results_dir"]) / ("fast" if args.fast else "")
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "metrics.csv", index=False)

    order = [m for m in MODELS if m in set(results.model)]
    summary = (results[results.nodes == "all"]
               .pivot_table(index=["target", "horizon_h"], columns="model", values="mae")
               [order].round(3))
    coverage = (results[results.nodes == "all"].groupby("model")[["coverage", "interval_width"]]
                .mean().reindex(order).round(3))
    by_nodes = (results.pivot_table(index="nodes", columns="model", values="mae")[order]
                .round(3))
    report = ("## MAE (log10), all nodes\n\n" + summary.to_markdown()
              + "\n\n## MAE (log10) by node group, averaged over targets and horizons\n\n"
              + by_nodes.to_markdown()
              + "\n\n## 80% interval (10-90% quantiles)\n\n" + coverage.to_markdown() + "\n")
    (out / "metrics.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
