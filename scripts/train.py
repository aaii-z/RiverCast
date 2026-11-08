import argparse
import time

from rivercast.experiment import load_setup, run_dir, save_run
from rivercast.models import MODELS, build_forecaster


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forecasters and save test predictions.")
    parser.add_argument("models", nargs="*", choices=MODELS, help="default: all")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--fast", action="store_true", help="smoke test: few batches/epochs")
    args = parser.parse_args()

    cfg, data = load_setup(args.config)
    train, val, test = (data.indices(s) for s in ("train", "val", "test"))
    print(f"origins: train {len(train)}, val {len(val)}, test {len(test)}; "
          f"{data.n_nodes} nodes, {data.n_features} features")

    for name in args.models or MODELS:
        print(f"\n== {name}")
        forecaster = build_forecaster(name, data, cfg, fast=args.fast)
        start = time.time()
        history = forecaster.fit(data, train, val)
        seconds = time.time() - start
        pred = data.to_log10(forecaster.predict(data, test))
        out = run_dir(cfg, name, args.fast)
        save_run(out, forecaster, test, pred, history,
                 {"model": name, "fast": args.fast, "train_seconds": round(seconds, 1)})
        print(f"   trained in {seconds:.0f}s -> {out}")


if __name__ == "__main__":
    main()
