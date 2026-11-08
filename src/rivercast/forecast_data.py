import numpy as np

from rivercast.dataset import Dataset


def forward_fill(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Last finite value along axis 0 and hours since it (NaN / inf before the first one)."""
    t = np.arange(len(a)).reshape(-1, *([1] * (a.ndim - 1)))
    last = np.where(np.isfinite(a), t, 0)
    np.maximum.accumulate(last, axis=0, out=last)
    filled = np.take_along_axis(a, last, axis=0)
    since = np.where(np.isfinite(filled), t - last, np.inf)
    return filled, since


def _standardize(x: np.ndarray, rows: slice, axis: tuple) -> tuple[np.ndarray, ...]:
    mean = np.nanmean(x[rows], axis=axis)
    std = np.maximum(np.nanstd(x[rows], axis=axis), 1e-6)
    return (x - mean) / std, mean, std


class ForecastData:
    """Model-ready view of a Dataset, standardized with training-split statistics.

    Node features per hour: dynamic inputs, observed targets (0 when missing) with
    observation masks, static attributes and time encodings. Targets are log10
    concentrations standardized per pathogen; `observed` is NaN where nothing was measured.
    """

    def __init__(self, ds: Dataset, window: int, horizons: list[int]):
        self.ds = ds
        self.window = window
        self.horizons = np.asarray(horizons)
        self.monitored = ds.nodes.monitored.to_numpy()
        train = ds.split("train")
        steps, nodes, _ = ds.target.shape

        dynamic = ds.dynamic.astype(np.float64)
        dynamic[..., 0] = np.log1p(dynamic[..., 0])
        dynamic[..., 1] = np.log(dynamic[..., 1])
        self.dynamic, _, _ = _standardize(dynamic, train, axis=(0, 1))

        self.observed, self.target_mean, self.target_std = _standardize(
            np.log10(ds.observed), train, axis=(0, 1))
        self.truth_log10 = np.log10(ds.target)

        static, _, _ = _standardize(ds.static, slice(None), axis=(0,))
        self.static = np.broadcast_to(static, (steps, *static.shape))

        hour = ds.times.hour.to_numpy() * 2 * np.pi / 24
        day = ds.times.dayofyear.to_numpy() * 2 * np.pi / 365.25
        time = np.column_stack([np.sin(hour), np.cos(hour), np.sin(day), np.cos(day)])
        self.time = np.broadcast_to(time[:, None, :], (steps, nodes, 4))

        mask = np.isfinite(self.observed)
        self.features = np.concatenate([
            self.dynamic, np.nan_to_num(self.observed), mask, self.static, self.time,
        ], axis=-1).astype(np.float32)

    @property
    def n_features(self) -> int:
        return self.features.shape[-1]

    @property
    def n_nodes(self) -> int:
        return self.features.shape[1]

    @property
    def n_targets(self) -> int:
        return self.observed.shape[-1]

    def indices(self, split: str) -> np.ndarray:
        """Forecast origins whose full input window and horizons lie inside the split."""
        s = self.ds.split(split)
        start = max(s.start or 0, self.window - 1)
        stop = (s.stop or len(self.features)) - self.horizons.max()
        return np.arange(start, stop)

    def windows(self, t: np.ndarray) -> np.ndarray:
        """Input windows [B, W, N, F] ending at each origin."""
        offsets = np.arange(-self.window + 1, 1)
        return self.features[t[:, None] + offsets]

    def targets(self, t: np.ndarray, truth: bool = False) -> np.ndarray:
        """Targets [B, N, H, P]: standardized observations, or true log10 values."""
        source = self.truth_log10 if truth else self.observed
        return source[t[:, None] + self.horizons].transpose(0, 2, 1, 3).astype(np.float32)

    def observed_log10(self, t: np.ndarray) -> np.ndarray:
        """Noisy, sparse observations [B, N, H, P] in log10 units (NaN where unobserved)."""
        return self.targets(t) * self.target_std + self.target_mean

    def to_log10(self, z: np.ndarray, targets: list[int] | None = None) -> np.ndarray:
        """Undo target scaling for predictions shaped [..., P, Q]."""
        k = slice(None) if targets is None else targets
        return z * self.target_std[k, None] + self.target_mean[k, None]
