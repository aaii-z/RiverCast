import copy
import time

import numpy as np
import torch
from torch import nn

from rivercast.forecast_data import ForecastData


def quantile_loss(pred: torch.Tensor, target: torch.Tensor,
                  quantiles: torch.Tensor) -> torch.Tensor:
    """Pinball loss on observed entries only; each target (task) weighted equally.

    pred [B, N, H, P, Q], target [B, N, H, P] with NaN where unobserved.
    """
    mask = torch.isfinite(target)
    diff = torch.nan_to_num(target).unsqueeze(-1) - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff).mean(-1) * mask
    count = mask.sum(dim=(0, 1, 2))
    per_task = loss.sum(dim=(0, 1, 2)) / count.clamp(min=1)
    return per_task[count > 0].mean()


class TorchForecaster:
    def __init__(self, module: nn.Module, quantiles: list[float], train_cfg: dict,
                 max_batches: int | None = None, targets: list[int] | None = None):
        self.module = module
        self.targets = targets  # train on a subset of targets (single-task runs)
        self.quantiles = tuple(quantiles)
        self.q = torch.tensor(quantiles, dtype=torch.float32)
        self.cfg = train_cfg
        self.max_batches = max_batches

    def _loss(self, data: ForecastData, t: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(data.windows(t))
        y = torch.from_numpy(data.targets(t))
        if self.targets is not None:
            y = y[..., self.targets]
        return quantile_loss(self.module(x), y, self.q)

    @torch.no_grad()
    def _evaluate(self, data: ForecastData, idx: np.ndarray, batch_size: int = 128) -> float:
        self.module.eval()
        losses = [self._loss(data, idx[i:i + batch_size]).item()
                  for i in range(0, len(idx), batch_size)]
        return float(np.mean(losses))

    def fit(self, data: ForecastData, train: np.ndarray, val: np.ndarray) -> list[dict]:
        cfg = self.cfg
        torch.manual_seed(cfg["seed"])
        rng = np.random.default_rng(cfg["seed"])
        opt = torch.optim.AdamW(self.module.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        best, best_state, stale, history = np.inf, None, 0, []

        for epoch in range(1, cfg["epochs"] + 1):
            start = time.time()
            self.module.train()
            order = rng.permutation(train)[:cfg["samples_per_epoch"]]
            batches = range(0, len(order), cfg["batch_size"])
            if self.max_batches:
                batches = batches[:self.max_batches]
            train_losses = []
            for i in batches:
                loss = self._loss(data, order[i:i + cfg["batch_size"]])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), cfg["grad_clip"])
                opt.step()
                train_losses.append(loss.item())

            val_loss = self._evaluate(data, val[::cfg["val_stride"]])
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                            "val_loss": val_loss, "seconds": time.time() - start})
            print(f"  epoch {epoch:2d}  train {history[-1]['train_loss']:.4f}  "
                  f"val {val_loss:.4f}  ({history[-1]['seconds']:.0f}s)")

            if val_loss < best:
                best, best_state, stale = val_loss, copy.deepcopy(self.module.state_dict()), 0
            else:
                stale += 1
                if stale >= cfg["patience"]:
                    print(f"  early stop (best val {best:.4f})")
                    break
        self.module.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def predict(self, data: ForecastData, t: np.ndarray, batch_size: int = 128) -> np.ndarray:
        self.module.eval()
        out = [self.module(torch.from_numpy(data.windows(t[i:i + batch_size]))).numpy()
               for i in range(0, len(t), batch_size)]
        return np.sort(np.concatenate(out), axis=-1)

    def save(self, path) -> None:
        torch.save(self.module.state_dict(), path / "model.pt")

    def load(self, path) -> None:
        self.module.load_state_dict(torch.load(path / "model.pt", weights_only=True))
