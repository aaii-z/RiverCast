import numpy as np
import pandas as pd


def forecast_metrics(pred_log10: np.ndarray, quantiles: tuple, truth_log10: np.ndarray,
                     monitored: np.ndarray, targets: list[str], horizons) -> pd.DataFrame:
    """Errors in log10 units against the true simulated concentration.

    pred [S, N, H, P, Q], truth [S, N, H, P]. Scores all nodes and, separately, nodes with
    and without sensors (unmonitored nodes test spatial inference).
    """
    median = pred_log10[..., quantiles.index(0.5)]
    lo, hi = pred_log10[..., 0], pred_log10[..., -1]
    interval = len(quantiles) > 1
    groups = {"all": np.ones_like(monitored), "monitored": monitored,
              "unmonitored": ~monitored}
    rows = []
    for group, nodes in groups.items():
        for k, target in enumerate(targets):
            for j, h in enumerate(horizons):
                truth = truth_log10[:, nodes, j, k]
                err = median[:, nodes, j, k] - truth
                row = {"target": target, "horizon_h": int(h), "nodes": group,
                       "mae": np.abs(err).mean(), "rmse": np.sqrt((err**2).mean()),
                       "bias": err.mean()}
                if interval:
                    lo_k, hi_k = lo[:, nodes, j, k], hi[:, nodes, j, k]
                    row["coverage"] = ((truth >= lo_k) & (truth <= hi_k)).mean()
                    row["interval_width"] = (hi_k - lo_k).mean()
                rows.append(row)
    return pd.DataFrame(rows)
