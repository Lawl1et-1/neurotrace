"""Spatial modules: message passing over the electrode graph of one window.

All modules map node features ``h`` (N, C, H) and an adjacency ``adj`` (N, C, C) in [src, dst]
to new node features (N, C, H).

* :class:`GATSpatial` - PyTorch Geometric ``GATConv``. Each node keeps its ``k`` strongest incoming
  edges; the edge weight enters the attention as an edge feature, so gradients reach the adjacency.
* :class:`DiffusionSpatial` - diffusion graph convolution (Li et al. 2018, DCRNN): forward and
  backward random walks of 1..K steps, which respects edge direction explicitly.
* :class:`NoGraphSpatial` - per-node MLP, the "ignore the graph" ablation.
"""

from __future__ import annotations

import torch
from torch import nn


def topk_edge_index(adj: torch.Tensor, k: int, src: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched dense adjacency -> PyG ``edge_index`` / ``edge_attr`` of one disjoint big graph.

    For each target node keep its ``k`` strongest sources (plus a self loop with weight 1). A given
    ``src`` (N, k, C) fixes which sources are kept; only their weights are read from ``adj``.
    """
    n, c, _ = adj.shape
    k = min(k, c - 1)
    a = adj.masked_fill(torch.eye(c, device=adj.device, dtype=torch.bool), -1.0)
    if src is None:
        vals, src = a.topk(k, dim=1)  # over sources, for every dst: (N, k, C)
    else:
        vals = a.gather(1, src)
    dst = torch.arange(c, device=adj.device).expand(n, k, c)
    offset = (torch.arange(n, device=adj.device) * c)[:, None, None]
    src_all = torch.cat([(src + offset).reshape(-1), torch.arange(n * c, device=adj.device)])
    dst_all = torch.cat([(dst + offset).reshape(-1), torch.arange(n * c, device=adj.device)])
    w = torch.cat([vals.reshape(-1), torch.ones(n * c, device=adj.device, dtype=adj.dtype)])
    return torch.stack([src_all, dst_all]), w[:, None]


class GATSpatial(nn.Module):
    """Stack of PyG ``GATConv`` layers with residual connections and layer norm."""

    def __init__(self, hidden: int, layers: int = 2, heads: int = 4, k: int = 6, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.k = k
        self.convs = nn.ModuleList(
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=1, dropout=dropout, add_self_loops=False)
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.last_attention: torch.Tensor | None = None
        # Integrated Gradients freezes the kept edges at the explained input: the top-k choice is a step
        # function of the input, and its jumps are invisible to gradients (the integral never converges).
        self.frozen_src: torch.Tensor | None = None

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        n, c, d = h.shape
        src = None
        if self.frozen_src is not None:
            src = self.frozen_src.repeat(n // self.frozen_src.shape[0], 1, 1)  # batch of paths, window-major
        else:
            k = min(self.k, c - 1)
            self.last_src = adj.masked_fill(torch.eye(c, device=adj.device, dtype=torch.bool), -1.0).topk(k, dim=1)[1].detach()
        edge_index, edge_attr = topk_edge_index(adj, self.k, src)
        x = h.reshape(n * c, d)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            out, (ei, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
            x = norm(x + torch.nn.functional.gelu(out))
        self.last_attention = alpha.detach().mean(dim=-1)  # (E,), last layer, mean over heads
        self.last_edge_index = ei.detach()
        return x.reshape(n, c, d)


class DiffusionSpatial(nn.Module):
    """Directed diffusion convolution: ``sum_k P_f^k H W_fk + P_b^k H W_bk``."""

    def __init__(self, hidden: int, layers: int = 2, steps: int = 2, dropout: float = 0.1):
        super().__init__()
        self.steps = steps
        self.lin = nn.ModuleList(nn.Linear((2 * steps + 1) * hidden, hidden) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # message from src to dst: H'_dst = sum_src P[src, dst] H_src  ->  H' = P^T H
        pf = adj / (adj.sum(dim=1, keepdim=True) + 1e-6)  # normalise incoming weights of each dst
        pb = adj.transpose(1, 2)
        pb = pb / (pb.sum(dim=1, keepdim=True) + 1e-6)
        x = h
        for lin, norm in zip(self.lin, self.norms, strict=True):
            feats = [x]
            xf, xb = x, x
            for _ in range(self.steps):
                xf = pf.transpose(1, 2) @ xf
                xb = pb.transpose(1, 2) @ xb
                feats += [xf, xb]
            x = norm(x + self.drop(torch.nn.functional.gelu(lin(torch.cat(feats, dim=-1)))))
        return x


class NoGraphSpatial(nn.Module):
    """Per-node MLP: no information flows between channels at this stage."""

    def __init__(self, hidden: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)) for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    def forward(self, h: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        for blk, norm in zip(self.blocks, self.norms, strict=True):
            h = norm(h + blk(h))
        return h
