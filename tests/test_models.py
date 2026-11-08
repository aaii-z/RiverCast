import numpy as np
import pytest
import torch

from rivercast.forecast_data import forward_fill
from rivercast.models.graph_wavenet import GraphWaveNet
from rivercast.models.lstm import NodeLSTM
from rivercast.training import quantile_loss

OUT = (4, 3, 3)  # horizons, targets, quantiles


def test_forward_fill():
    a = np.array([np.nan, 1.0, np.nan, np.nan, 5.0])[:, None]
    filled, since = forward_fill(a)
    assert np.isnan(filled[0, 0]) and since[0, 0] == np.inf
    np.testing.assert_array_equal(filled[1:, 0], [1, 1, 1, 5])
    np.testing.assert_array_equal(since[1:, 0], [0, 1, 2, 0])


def test_quantile_loss_ignores_missing_and_weights_tasks_equally():
    q = torch.tensor([0.1, 0.5, 0.9])
    pred = torch.zeros(2, 3, 1, 2, 3)
    target = torch.full((2, 3, 1, 2), float("nan"))
    target[0, 0, 0, 0] = 1.0                  # task 0: one label, error 1
    target[:, :, 0, 1] = 2.0                  # task 1: six labels, error 2
    expected = (q.mean() * 1 + q.mean() * 2) / 2
    assert torch.isclose(quantile_loss(pred, target, q), expected)


@pytest.mark.parametrize("model", ["lstm", "gwn"])
def test_output_shape(model):
    nodes, features = 5, 7
    adjacency = np.eye(nodes, k=1, dtype=np.float32)  # chain 0 -> 1 -> ... -> 4
    if model == "lstm":
        net = NodeLSTM(features, OUT, hidden=8, layers=1, dropout=0.0)
    else:
        net = GraphWaveNet(adjacency, features, OUT, channels=8, skip_channels=8,
                           end_channels=8, blocks=1, layers=3, kernel=2, order=2,
                           embed_dim=4, dropout=0.0)
    assert net(torch.randn(2, 12, nodes, features)).shape == (2, nodes, *OUT)

