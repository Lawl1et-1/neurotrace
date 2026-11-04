"""The NEUROTRACE model.

Pipeline for a sequence of S windows (each C channels x L samples):

1. **Node encoder** - a small 1D CNN per channel (shared weights) on the raw window, plus a linear
   map of its band powers, plus a learned electrode embedding.
2. **Graph** - ``none`` (no message passing), ``static`` (distance prior), or ``learned``:
   ``A_t = mix * A_prior + (1 - mix) * A_learned_t`` where ``A_learned_t`` is the directed,
   per-window adjacency of :class:`~neurotrace.graph.learned.DirectedEdgeInference`.
3. **Spatial GNN** - GAT or diffusion convolution over ``A_t`` (or a per-node MLP for ``none``).
4. **Readout** - a learned projection of all node embeddings to one graph embedding per window.
5. **Temporal** - a Transformer over the S graph embeddings (or a per-window MLP for ``none``).
6. **Heads** - band power of the *next* window (``objective=next``) or of masked windows
   (``objective=masked``), plus an optional supervised seizure logit.

Every switch above is a config value so the ablations use exactly the same code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from neurotrace.graph.learned import DirectedEdgeInference
from neurotrace.graph.prior import distance_prior
from neurotrace.models.spatial import DiffusionSpatial, GATSpatial, NoGraphSpatial
from neurotrace.models.temporal import NoTemporal, TemporalTransformer


@dataclass
class ModelConfig:
    """Plain mirror of ``configs/model/neurotrace.yaml`` (plus data-dependent sizes)."""

    n_channels: int
    window_len: int
    n_bands: int = 5
    graph: str = "learned"  # none | static | learned
    spatial: str = "gat"  # gat | diffusion
    temporal: str = "transformer"  # transformer | none
    input: str = "both"  # raw | bands | both
    hidden: int = 64
    gnn_layers: int = 2
    heads: int = 4
    tf_layers: int = 2
    tf_heads: int = 4
    dropout: float = 0.1
    seq_len: int = 8
    causal: bool = True
    prior_sigma: float = 0.5
    prior_mix: float = 0.5
    max_lag: int = 8
    var_order: int = 5  # 0 disables the conditional VAR evidence (ablation)
    detach_graph: bool = True  # the GNN reads the learned graph but does not train it (DECISIONS D6)
    objective: str = "next"
    supervised: bool = False

    @classmethod
    def from_cfg(cls, model_cfg, train_cfg, n_channels: int, window_len: int, n_bands: int) -> ModelConfig:
        keys = cls.__dataclass_fields__.keys()
        kw = {k: v for k, v in dict(model_cfg).items() if k in keys}
        return cls(
            n_channels=n_channels,
            window_len=window_len,
            n_bands=n_bands,
            objective=train_cfg.objective,
            supervised=train_cfg.supervised,
            **{k: v for k, v in kw.items() if k not in ("n_channels", "window_len", "n_bands")},
        )


class NodeEncoder(nn.Module):
    """Raw window + band powers + electrode identity -> node embedding (N, C, H)."""

    def __init__(self, n_channels: int, n_bands: int, hidden: int, mode: str = "both"):
        super().__init__()
        self.mode = mode
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(32, hidden, 5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.band = nn.Linear(n_bands, hidden)
        self.chan = nn.Parameter(torch.randn(n_channels, hidden) * 0.02)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        n, c, length = x.shape
        h = self.chan[None].expand(n, -1, -1)
        if self.mode in ("raw", "both"):
            h = h + self.cnn(x.reshape(n * c, 1, length)).reshape(n, c, -1)
        if self.mode in ("bands", "both"):
            h = h + self.band(bands)
        return self.norm(h)


class NeuroTrace(nn.Module):
    """Spatio-temporal graph model; see the module docstring."""

    def __init__(self, cfg: ModelConfig, ch_names: list[str]):
        super().__init__()
        self.cfg = cfg
        c, h = cfg.n_channels, cfg.hidden
        self.encoder = NodeEncoder(c, cfg.n_bands, h, cfg.input)
        self.register_buffer("prior", torch.from_numpy(distance_prior(ch_names, cfg.prior_sigma)))
        self.register_buffer("band_mean", torch.zeros(c, cfg.n_bands))
        self.register_buffer("band_std", torch.ones(c, cfg.n_bands))

        self.edges = DirectedEdgeInference(h, cfg.max_lag, dropout=cfg.dropout, var_order=cfg.var_order) if cfg.graph == "learned" else None
        if cfg.graph == "none":
            self.spatial: nn.Module = NoGraphSpatial(h, cfg.gnn_layers, cfg.dropout)
        elif cfg.spatial == "gat":
            self.spatial = GATSpatial(h, cfg.gnn_layers, cfg.heads, dropout=cfg.dropout)
        elif cfg.spatial == "diffusion":
            self.spatial = DiffusionSpatial(h, cfg.gnn_layers, dropout=cfg.dropout)
        else:
            raise ValueError(f"unknown spatial module {cfg.spatial!r}")

        d = 2 * h
        self.readout = nn.Sequential(nn.Linear(c * h, d), nn.GELU(), nn.LayerNorm(d))
        if cfg.temporal == "transformer":
            causal = cfg.causal or cfg.objective == "next"  # next-window prediction must not peek ahead
            self.temporal: nn.Module = TemporalTransformer(d, cfg.tf_layers, cfg.tf_heads, cfg.dropout, causal)
        else:
            self.temporal = NoTemporal(d, cfg.dropout)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c * cfg.n_bands))
        self.cls = nn.Linear(d, 1) if cfg.supervised else None

    # ------------------------------------------------------------------ per-window part
    def set_band_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Store training-set band-power statistics used to standardise inputs and targets."""
        self.band_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.band_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def norm_bands(self, bands: torch.Tensor) -> torch.Tensor:
        return (bands - self.band_mean) / self.band_std

    def encode_windows(self, x: torch.Tensor, bands: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process N independent windows: x (N, C, L), standardised bands (N, C, F).

        Returns the graph embedding ``z`` (N, D), the adjacency used by the GNN ``adj_gnn``,
        the learned part ``adj`` (or None), node embeddings and the within-window forecast error.
        """
        n = x.shape[0]
        h = self.encoder(x, bands)
        out: dict[str, torch.Tensor] = {}
        if self.cfg.graph == "learned":
            e = self.edges(h, x)
            pred, target = self.edges.forecast(x, e)
            out["adj"] = e["adj"]
            out["var_err"] = ((pred - target) ** 2).mean(dim=(1, 2))  # (N,)
            mix = self.cfg.prior_mix
            learned = e["adj"].detach() if self.cfg.detach_graph else e["adj"]
            adj_gnn = mix * self.prior[None] + (1 - mix) * learned
        elif self.cfg.graph == "static":
            adj_gnn = self.prior[None].expand(n, -1, -1)
        else:
            adj_gnn = None
        hs = self.spatial(h, adj_gnn) if adj_gnn is not None else self.spatial(h)
        out["nodes"] = hs
        out["z"] = self.readout(hs.reshape(n, -1))
        if adj_gnn is not None:
            out["adj_gnn"] = adj_gnn
        return out

    # ------------------------------------------------------------------ sequence part
    def temporal_forward(self, z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """z (B, S, D) -> band predictions (B, S, C, F) and optional seizure logits (B, S).

        For ``objective=next`` position t predicts window t + 1; for ``masked`` it predicts window t
        (the masked positions are replaced by a learned token first).
        """
        if mask is not None:
            z = torch.where(mask[..., None], self.mask_token.expand_as(z), z)
        y = self.temporal(z)
        b, s, _ = y.shape
        out = {"pred": self.head(y).reshape(b, s, self.cfg.n_channels, self.cfg.n_bands), "y": y}
        if self.cls is not None:
            out["logits"] = self.cls(y).squeeze(-1)
        return out

    def forward(self, x: torch.Tensor, bands: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """x (B, S, C, L), raw bands (B, S, C, F)."""
        b, s, c, length = x.shape
        nb = self.norm_bands(bands)
        enc = self.encode_windows(x.reshape(b * s, c, length), nb.reshape(b * s, c, -1))
        out = self.temporal_forward(enc["z"].reshape(b, s, -1), mask)
        out["bands_n"] = nb
        for k in ("adj", "adj_gnn", "var_err"):
            if k in enc:
                out[k] = enc[k].reshape(b, s, *enc[k].shape[1:])
        return out
