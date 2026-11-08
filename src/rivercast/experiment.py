import json
from pathlib import Path

import numpy as np
import pandas as pd

from rivercast.config import load_config
from rivercast.dataset import Dataset
from rivercast.forecast_data import ForecastData
from rivercast.models import build_forecaster


def load_setup(config_path: str) -> tuple[dict, ForecastData]:
    cfg = load_config(config_path)
    data_cfg = load_config(cfg["data_config"])
    ds = Dataset.load(Path(data_cfg["paths"]["processed"]))
    return cfg, ForecastData(ds, cfg["window_h"], cfg["horizons_h"])


def run_dir(cfg: dict, name: str, fast: bool = False, target: str | None = None) -> Path:
    stem = f"{name}_{target}" if target else name
    return Path(cfg["runs_dir"]) / (f"{stem}_fast" if fast else stem)


def save_run(path: Path, forecaster, origins: np.ndarray, pred_log10: np.ndarray,
             history: list[dict], info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "predictions.npz", origins=origins,
                        pred_log10=pred_log10.astype(np.float32),
                        quantiles=np.array(forecaster.quantiles))
    pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    (path / "run.json").write_text(json.dumps(info, indent=2))
    forecaster.save(path)


def load_predictions(path: Path, file: str = "predictions.npz"):
    p = np.load(path / file)
    return p["origins"], p["pred_log10"], tuple(float(q) for q in p["quantiles"])


def predict_split(cfg: dict, data: ForecastData, name: str, split: str,
                  target: str | None = None) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Predictions of a trained run on a split; test comes from training, others are cached."""
    path = run_dir(cfg, name, target=target)
    if split == "test":
        return load_predictions(path)
    cache = path / f"{split}_predictions.npz"
    if not cache.exists():
        targets = [data.ds.meta["targets"].index(target)] if target else None
        forecaster = build_forecaster(name, data, cfg, targets=targets)
        forecaster.load(path)
        origins = data.indices(split)
        pred = data.to_log10(forecaster.predict(data, origins), targets)
        np.savez_compressed(cache, origins=origins, pred_log10=pred.astype(np.float32),
                            quantiles=np.array(forecaster.quantiles))
    return load_predictions(path, cache.name)


def predict_single_task(cfg: dict, data: ForecastData, name: str, split: str):
    """Stack per-pathogen single-task runs into one [S, N, H, P, Q] prediction."""
    parts = [predict_split(cfg, data, name, split, target) for target in data.ds.meta["targets"]]
    origins, _, quantiles = parts[0]
    return origins, np.concatenate([p[1] for p in parts], axis=3), quantiles
