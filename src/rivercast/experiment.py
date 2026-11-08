import json
from pathlib import Path

import numpy as np
import pandas as pd

from rivercast.config import load_config
from rivercast.dataset import Dataset
from rivercast.forecast_data import ForecastData


def load_setup(config_path: str) -> tuple[dict, ForecastData]:
    cfg = load_config(config_path)
    data_cfg = load_config(cfg["data_config"])
    ds = Dataset.load(Path(data_cfg["paths"]["processed"]))
    return cfg, ForecastData(ds, cfg["window_h"], cfg["horizons_h"])


def run_dir(cfg: dict, name: str, fast: bool) -> Path:
    return Path(cfg["runs_dir"]) / (f"{name}_fast" if fast else name)


def save_run(path: Path, forecaster, origins: np.ndarray, pred_log10: np.ndarray,
             history: list[dict], info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "predictions.npz", origins=origins,
                        pred_log10=pred_log10.astype(np.float32),
                        quantiles=np.array(forecaster.quantiles))
    pd.DataFrame(history).to_csv(path / "history.csv", index=False)
    (path / "run.json").write_text(json.dumps(info, indent=2))
    forecaster.save(path)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, tuple]:
    p = np.load(path / "predictions.npz")
    return p["origins"], p["pred_log10"], tuple(float(q) for q in p["quantiles"])
