import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ARRAYS = ("dynamic", "static", "target", "observed", "adjacency")


@dataclass
class Dataset:
    """Hourly river dataset; arrays are [time, node, feature]."""

    times: pd.DatetimeIndex
    dynamic: np.ndarray
    static: np.ndarray
    target: np.ndarray
    observed: np.ndarray
    adjacency: np.ndarray
    nodes: pd.DataFrame
    meta: dict

    def split(self, name: str) -> slice:
        val, test = (pd.Timestamp(self.meta["split"][k]) for k in ("val_start", "test_start"))
        bounds = {"train": (None, val), "val": (val, test), "test": (test, None)}[name]
        start, stop = (None if b is None else self.times.searchsorted(b) for b in bounds)
        return slice(start, stop)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "dataset.npz", times=self.times.to_numpy(),
                            **{k: getattr(self, k).astype(np.float32) for k in ARRAYS})
        self.nodes.to_csv(out_dir / "nodes.csv", index=False)
        (out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=str))

    @classmethod
    def load(cls, out_dir: Path) -> "Dataset":
        data = np.load(out_dir / "dataset.npz")
        return cls(times=pd.DatetimeIndex(data["times"]),
                   **{k: data[k] for k in ARRAYS},
                   nodes=pd.read_csv(out_dir / "nodes.csv"),
                   meta=json.loads((out_dir / "meta.json").read_text()))
