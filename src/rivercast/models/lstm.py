import torch
from torch import nn


class NodeLSTM(nn.Module):
    """Temporal-only baseline: one LSTM shared across nodes, no information between nodes."""

    def __init__(self, n_features: int, out_shape: tuple[int, int, int], hidden: int,
                 layers: int, dropout: float):
        super().__init__()
        self.out_shape = out_shape
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, out_shape[0] * out_shape[1] * out_shape[2]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, W, N, F] -> [B, N, H, P, Q]."""
        b, w, n, f = x.shape
        _, (hidden, _) = self.lstm(x.permute(0, 2, 1, 3).reshape(b * n, w, f))
        return self.head(hidden[-1]).view(b, n, *self.out_shape)
