from rivercast.forecast_data import ForecastData
from rivercast.models.baselines import Persistence
from rivercast.models.graph_wavenet import GraphWaveNet
from rivercast.models.lstm import NodeLSTM
from rivercast.models.xgb import XGBForecaster
from rivercast.training import TorchForecaster

MODELS = ("persistence", "xgboost", "lstm", "graph_wavenet")


def build_forecaster(name: str, data: ForecastData, cfg: dict, fast: bool = False):
    quantiles = cfg["quantiles"]
    out_shape = (len(data.horizons), data.n_targets, len(quantiles))
    train_cfg = {**cfg["train"], **({"epochs": cfg["fast"]["epochs"]} if fast else {})}
    max_batches = cfg["fast"]["max_batches"] if fast else None

    if name == "persistence":
        return Persistence()
    if name == "xgboost":
        params = dict(cfg["xgboost"])
        max_rows = cfg["fast"]["xgb_rows"] if fast else params.pop("max_rows")
        params.pop("max_rows", None)
        if fast:
            params["n_estimators"] = 50
        return XGBForecaster(quantiles, params, max_rows, cfg["train"]["seed"])
    if name == "lstm":
        module = NodeLSTM(data.n_features, out_shape, **cfg["lstm"])
    elif name == "graph_wavenet":
        module = GraphWaveNet(data.ds.adjacency, data.n_features, out_shape,
                              **cfg["graph_wavenet"])
    else:
        raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")
    return TorchForecaster(module, quantiles, train_cfg, max_batches)
