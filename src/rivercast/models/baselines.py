import numpy as np

from rivercast.forecast_data import ForecastData, forward_fill


class Persistence:
    """Last observed value at each node, or the training mean where nothing was observed yet."""

    quantiles = (0.5,)

    def fit(self, data: ForecastData, train: np.ndarray, val: np.ndarray) -> list[dict]:
        self.last, _ = forward_fill(data.observed)
        return []

    def predict(self, data: ForecastData, t: np.ndarray) -> np.ndarray:
        last = np.nan_to_num(self.last[t], nan=0.0)
        shape = (len(t), data.n_nodes, len(data.horizons), data.n_targets, 1)
        return np.broadcast_to(last[:, :, None, :, None], shape).copy()

    def save(self, path) -> None:
        pass
