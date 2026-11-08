import torch
import torch.nn.functional as F
from torch import nn


def _row_normalize(a: torch.Tensor) -> torch.Tensor:
    return a / a.sum(dim=1, keepdim=True).clamp(min=1.0)


class DiffusionConv(nn.Module):
    """K-step diffusion over several supports, mixed by a 1x1 convolution (Wu et al., 2019)."""

    def __init__(self, c_in: int, c_out: int, n_supports: int, order: int, dropout: float):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mix = nn.Conv2d(c_in * (1 + n_supports * order), c_out, kernel_size=1)

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        """x [B, C, N, T]; support[v, w] > 0 means information flows from node v to w."""
        out = [x]
        for a in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("bcvt,vw->bcwt", h, a)
                out.append(h)
        return F.dropout(self.mix(torch.cat(out, dim=1)), self.dropout, self.training)


class GraphWaveNet(nn.Module):
    """Graph WaveNet: gated dilated temporal convolutions + diffusion graph convolutions
    over the river's downstream/upstream transitions and a learned adaptive adjacency."""

    def __init__(self, adjacency, n_features: int, out_shape: tuple[int, int, int],
                 channels: int, skip_channels: int, end_channels: int, blocks: int, layers: int,
                 kernel: int, order: int, embed_dim: int, dropout: float):
        super().__init__()
        a = torch.as_tensor(adjacency, dtype=torch.float32)
        n = a.shape[0]
        self.out_shape = out_shape
        self.register_buffer("downstream", _row_normalize(a))
        self.register_buffer("upstream", _row_normalize(a.T))
        self.source = nn.Parameter(0.1 * torch.randn(n, embed_dim))
        self.target = nn.Parameter(0.1 * torch.randn(n, embed_dim))

        dilations = [2**i for _ in range(blocks) for i in range(layers)]
        self.receptive_field = 1 + (kernel - 1) * sum(dilations)
        self.start = nn.Conv2d(n_features, channels, kernel_size=1)
        self.filters = nn.ModuleList(
            nn.Conv2d(channels, channels, (1, kernel), dilation=(1, d)) for d in dilations)
        self.gates = nn.ModuleList(
            nn.Conv2d(channels, channels, (1, kernel), dilation=(1, d)) for d in dilations)
        self.skips = nn.ModuleList(nn.Conv2d(channels, skip_channels, 1) for _ in dilations)
        self.graph_convs = nn.ModuleList(
            DiffusionConv(channels, channels, 3, order, dropout) for _ in dilations)
        self.norms = nn.ModuleList(nn.BatchNorm2d(channels) for _ in dilations)
        self.end = nn.Sequential(
            nn.ReLU(), nn.Conv2d(skip_channels, end_channels, 1),
            nn.ReLU(), nn.Conv2d(end_channels, out_shape[0] * out_shape[1] * out_shape[2], 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, W, N, F] -> [B, N, H, P, Q]."""
        b, w, n, _ = x.shape
        x = x.permute(0, 3, 2, 1)
        if w < self.receptive_field:
            x = F.pad(x, (self.receptive_field - w, 0))
        x = self.start(x)

        adaptive = F.softmax(F.relu(self.source @ self.target.T), dim=1)
        supports = [self.downstream, self.upstream, adaptive]
        skip = 0
        for filt, gate, skip_conv, graph_conv, norm in zip(
                self.filters, self.gates, self.skips, self.graph_convs, self.norms, strict=True):
            residual = x
            x = torch.tanh(filt(x)) * torch.sigmoid(gate(x))
            skip = skip + skip_conv(x[..., -1:])
            x = norm(graph_conv(x, supports) + residual[..., -x.shape[-1]:])

        out = self.end(skip).squeeze(-1)  # [B, H*P*Q, N]
        return out.permute(0, 2, 1).reshape(b, n, *self.out_shape)
