import numpy as np
import pandas as pd


def conformal_offsets(pred: np.ndarray, observed: np.ndarray, coverage: float) -> np.ndarray:
    """Conformalized quantile regression (Romano et al., 2019).

    pred [S, N, H, P, Q] and observed [S, N, H, P] (log10, NaN where unobserved) come from a
    held-out calibration period. Returns the widening [H, P] that makes the outer quantile
    band cover `coverage` of calibration observations.
    """
    score = np.maximum(pred[..., 0] - observed, observed - pred[..., -1])
    _, _, horizons, targets = score.shape
    offsets = np.zeros((horizons, targets))
    for j in range(horizons):
        for k in range(targets):
            s = score[..., j, k]
            s = s[np.isfinite(s)]
            level = min(1.0, np.ceil((len(s) + 1) * coverage) / len(s))
            offsets[j, k] = np.quantile(s, level, method="higher")
    return offsets


def widen(pred: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    out = pred.copy()
    out[..., 0] -= offsets
    out[..., -1] += offsets
    return out


def coverage(pred: np.ndarray, reference: np.ndarray) -> pd.DataFrame:
    """Share of finite reference values inside the outer band, per target and overall."""
    inside = (reference >= pred[..., 0]) & (reference <= pred[..., -1])
    valid = np.isfinite(reference)
    per_target = (inside & valid).sum(axis=(0, 1, 2)) / valid.sum(axis=(0, 1, 2))
    width = np.where(valid, pred[..., -1] - pred[..., 0], np.nan)
    return pd.DataFrame({"coverage": per_target,
                         "width": np.nanmean(width, axis=(0, 1, 2))})


def high_pollution_mask(truth: np.ndarray, reference: np.ndarray, pct: float = 95) -> np.ndarray:
    """True where the true value exceeds each node and target's `pct` percentile in
    `reference` (truth over the training period, [T, N, P])."""
    threshold = np.percentile(reference, pct, axis=0)  # [N, P]
    return truth > threshold[None, :, None, :]


def mae_by_regime(median: np.ndarray, truth: np.ndarray, high: np.ndarray,
                  targets: list[str]) -> pd.DataFrame:
    err = np.abs(median - truth)
    rows = []
    for k, name in enumerate(targets):
        h = high[..., k]
        rows.append({"target": name, "calm": err[..., k][~h].mean(),
                     "high_pollution": err[..., k][h].mean(), "share_high": h.mean()})
    return pd.DataFrame(rows)


def alert_skill(predicted_log10: np.ndarray, truth: np.ndarray, threshold_log10: float) -> dict:
    """Exceedance alerts treated as a binary forecast."""
    alert, event = predicted_log10 > threshold_log10, truth > threshold_log10
    hits = (alert & event).sum()
    precision = hits / max(alert.sum(), 1)
    recall = hits / max(event.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "events": int(event.sum())}
