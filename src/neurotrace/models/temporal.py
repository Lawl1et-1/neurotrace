"""Temporal module: a Transformer encoder over the sequence of per-window graph embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.pe[: z.shape[1]][None]


class TemporalTransformer(nn.Module):
    """Pre-norm Transformer encoder. With ``causal=True`` position t only attends to positions <= t.

    The attention weights of the last forward pass are kept in ``last_attention``
    (layers, B, heads, S, S) for the explainability views.
    """

    def __init__(self, dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.1, causal: bool = True):
        super().__init__()
        self.causal = causal
        self.pe = SinusoidalPE(dim)
        self.layers = nn.ModuleList(_Block(dim, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.last_attention: torch.Tensor | None = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        s = z.shape[1]
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(s, s, device=z.device, dtype=torch.bool), diagonal=1)
        x = self.pe(z)
        atts = []
        for blk in self.layers:
            x, att = blk(x, mask)
            atts.append(att)
        self.last_attention = torch.stack(atts).detach()
        return self.norm(x)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None):
        y = self.n1(x)
        y, att = self.att(y, y, y, attn_mask=mask, need_weights=True, average_attn_weights=False)
        x = x + self.drop(y)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, att


class NoTemporal(nn.Module):
    """Ablation: each window is processed on its own (an MLP instead of attention over time)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.last_attention = None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.mlp(z))
