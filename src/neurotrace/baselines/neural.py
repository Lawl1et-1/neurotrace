"""Neural anomaly baselines that model each channel on its own.

:class:`ChannelCNNLSTM` encodes every window of every channel with the same small 1D CNN, runs an
LSTM over time *per channel* and predicts that channel's next-window band power. No information
flows between channels, so comparing it with NEUROTRACE measures what the graph adds.
It exposes the same ``encode/forward`` interface pieces used by the training loop
(``cfg``, ``norm_bands``, ``set_band_stats``) so it can be trained and scored with the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ChannelBaselineConfig:
    n_channels: int
    window_len: int
    n_bands: int = 5
    hidden: int = 64
    seq_len: int = 8
    objective: str = "next"
    graph: str = "none"


class ChannelCNNLSTM(nn.Module):
    """Per-channel CNN + LSTM next-window forecaster (shared weights across channels)."""

    def __init__(self, cfg: ChannelBaselineConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(16, h, 5, stride=2, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )  # fmt: skip
        self.band = nn.Linear(cfg.n_bands, h)
        self.lstm = nn.LSTM(h, h, batch_first=True)
        self.head = nn.Linear(h, cfg.n_bands)
        self.cls = None
        self.register_buffer("band_mean", torch.zeros(cfg.n_channels, cfg.n_bands))
        self.register_buffer("band_std", torch.ones(cfg.n_channels, cfg.n_bands))

    def set_band_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.band_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.band_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def norm_bands(self, bands: torch.Tensor) -> torch.Tensor:
        return (bands - self.band_mean) / self.band_std

    def forward(self, x: torch.Tensor, bands: torch.Tensor, mask=None) -> dict[str, torch.Tensor]:
        b, s, c, length = x.shape
        nb = self.norm_bands(bands)
        e = self.cnn(x.reshape(b * s * c, 1, length)).reshape(b, s, c, -1) + self.band(nb)
        e = e.permute(0, 2, 1, 3).reshape(b * c, s, -1)  # one sequence per channel
        y, _ = self.lstm(e)
        pred = self.head(y).reshape(b, c, s, -1).permute(0, 2, 1, 3)
        return {"pred": pred, "bands_n": nb}


@torch.no_grad()
def channel_baseline_errors(model: ChannelCNNLSTM, rec, device: str = "cpu", chunk: int = 128) -> np.ndarray:
    """Next-window band-power error per window (NaN for the first ``seq_len - 1`` windows)."""
    model.eval().to(device)
    s, n = model.cfg.seq_len, len(rec)
    err = np.full(n, np.nan, dtype=np.float32)
    if n < s:
        return err
    xw = torch.as_tensor(np.asarray(rec.windows, dtype=np.float32))
    bw = torch.as_tensor(rec.bands)
    for i in range(0, n - s + 1, chunk):
        idx = torch.arange(i, min(i + chunk, n - s + 1))
        win = idx[:, None] + torch.arange(s)[None]
        out = model(xw[win].to(device), bw[win].to(device))
        e = ((out["pred"][:, -2] - out["bands_n"][:, -1]) ** 2).mean(dim=(1, 2))
        err[(idx + s - 1).numpy()] = e.cpu().numpy()
    return err
