import argparse
from pathlib import Path

import pandas as pd

from rivercast.config import load_config
from rivercast.dataset import Dataset
from rivercast.plots import largest_rain_event, plot_event, plot_network, plot_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the network and a storm event.")
    parser.add_argument("--config", default="configs/norwich.yaml")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    data_dir = Path(load_config(args.config)["paths"]["processed"])
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    ds = Dataset.load(data_dir)
    event = largest_rain_event(ds)
    print(f"largest 24 h rain event starts {event}")

    plot_network(ds, data_dir / "rivers.geojson").figure.savefig(
        out / "network.png", dpi=160, bbox_inches="tight")
    plot_event(ds, event - pd.Timedelta(hours=12)).savefig(
        out / "event.png", dpi=160, bbox_inches="tight")
    plot_observations(ds, "YAR13", event - pd.Timedelta(days=7)).savefig(
        out / "observations.png", dpi=160, bbox_inches="tight")
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
