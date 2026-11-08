import argparse

from rivercast.config import load_config
from rivercast.pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch inputs and simulate the river dataset.")
    parser.add_argument("--config", default="configs/norwich.yaml")
    ds = build_dataset(load_config(parser.parse_args().config))
    t, n, p = ds.target.shape
    print(f"saved: {t} hours x {n} nodes x {p} targets ({ds.times[0]} -> {ds.times[-1]})")


if __name__ == "__main__":
    main()
