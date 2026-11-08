import json

import numpy as np
import xgboost as xgb

from rivercast.forecast_data import ForecastData, forward_fill


def tabular_features(data: ForecastData) -> np.ndarray:
    """Per-node engineered features [T, N, F] using only that node's own history."""
    rain = data.ds.dynamic[..., 0]
    log_flow = np.log(data.ds.dynamic[..., 1])
    cum = np.cumsum(rain, axis=0)

    def rain_sum(h):
        return np.log1p(cum - np.concatenate([np.zeros((h, rain.shape[1])), cum[:-h]]))

    def flow_change(h):
        return log_flow - np.concatenate([np.repeat(log_flow[:1], h, axis=0), log_flow[:-h]])

    last, since = forward_fill(data.observed)
    engineered = np.stack([rain_sum(3), rain_sum(6), rain_sum(24),
                           flow_change(3), flow_change(24)], axis=-1)
    return np.concatenate([data.dynamic, engineered, last, np.minimum(since, 336) / 24,
                           data.static, data.time], axis=-1).astype(np.float32)


class XGBForecaster:
    """Non-graph gradient boosting: one multi-quantile model per target and horizon."""

    def __init__(self, quantiles: list[float], params: dict, max_rows: int, seed: int):
        self.quantiles = tuple(quantiles)
        self.params = params
        self.max_rows = max_rows
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.models: dict[tuple[int, int], xgb.Booster] = {}

    def _rows(self, x, data, t, h, k, max_rows):
        y = data.observed[t + h, :, k]
        ok = np.isfinite(y)
        xs, ys = x[t][ok], y[ok]
        if len(ys) > max_rows:
            pick = self.rng.choice(len(ys), max_rows, replace=False)
            xs, ys = xs[pick], ys[pick]
        return xs, ys

    def fit(self, data: ForecastData, train: np.ndarray, val: np.ndarray) -> list[dict]:
        x = tabular_features(data)
        history = []
        for k in range(data.n_targets):
            for j, h in enumerate(data.horizons):
                x_tr, y_tr = self._rows(x, data, train, h, k, self.max_rows)
                x_va, y_va = self._rows(x, data, val, h, k, self.max_rows // 4)
                params = {k_: v for k_, v in self.params.items() if k_ != "n_estimators"}
                params |= {"objective": "reg:quantileerror", "tree_method": "hist",
                           "quantile_alpha": np.array(self.quantiles), "seed": self.seed}
                booster = xgb.train(params, xgb.DMatrix(x_tr, y_tr),
                                    num_boost_round=self.params["n_estimators"],
                                    evals=[(xgb.DMatrix(x_va, y_va), "val")],
                                    early_stopping_rounds=30, verbose_eval=False)
                self.models[(k, j)] = booster
                history.append({"target": k, "horizon": int(h), "rows": len(y_tr),
                                "best_iteration": booster.best_iteration,
                                "val_loss": booster.best_score})
                print(f"  target {k} horizon {h}h: {len(y_tr)} rows, "
                      f"{booster.best_iteration + 1} trees, val pinball {booster.best_score:.4f}")
        return history

    def predict(self, data: ForecastData, t: np.ndarray) -> np.ndarray:
        x = tabular_features(data)[t]
        flat = xgb.DMatrix(x.reshape(-1, x.shape[-1]))
        out = np.empty((len(t), data.n_nodes, len(data.horizons), data.n_targets,
                        len(self.quantiles)), dtype=np.float32)
        for (k, j), model in self.models.items():
            best = (0, model.best_iteration + 1)
            pred = model.predict(flat, iteration_range=best)
            out[:, :, j, k, :] = pred.reshape(len(t), data.n_nodes, -1)
        return out

    def save(self, path) -> None:
        for (k, j), model in self.models.items():
            model.save_model(path / f"xgb_t{k}_h{j}.json")
        (path / "xgb_params.json").write_text(json.dumps(self.params, indent=2))

    def load(self, path) -> None:
        for file in path.glob("xgb_t*_h*.json"):
            k, j = (int(part[1:]) for part in file.stem.split("_")[1:])
            booster = xgb.Booster()
            booster.load_model(file)
            self.models[(k, j)] = booster
