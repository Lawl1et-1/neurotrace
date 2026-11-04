"""Learnable, directed, time-varying adjacency.

For every window the module infers a directed graph ``A_t[src, dst]`` in [0, 1] and is trained by
asking that graph to *explain the window*: each channel's next sample is forecast from its own past
plus the gated, lagged past of the other channels (an amortised neural VAR, in the spirit of
amortised causal discovery, Loewe et al. 2022). An edge only earns a large gate if the source's past
helps forecast the target, which is a Granger-style, directed statistical dependency - not proof of
causation (DECISIONS D2).

Edge scores combine four things:
* the source and target node embeddings (what each channel is doing in this window),
* lagged cross-correlations ``src(t - k) x dst(t)`` for k = 0..K (direction-aware evidence),
* *conditional* evidence from a ridge VAR fitted to the window in closed form: the Wald statistic of
  each ``src -> dst`` lag block and the coefficients themselves. Pairwise correlations cannot tell a
  direct coupling from a shared input; conditioning on all other channels can (DECISIONS D6),
* the same features in the reverse direction, so the MLP can compare ``i -> j`` with ``j -> i``.
"""

from __future__ import annotations

import torch
from torch import nn


def lagged_xcorr(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """Normalised lagged cross-correlation.

    ``x``: (N, C, L), each channel roughly zero-mean. Returns (N, C, C, max_lag + 1) where
    ``out[n, s, d, k] = corr(x_s[t - k], x_d[t])``, i.e. evidence that ``s`` leads ``d`` by ``k``.
    """
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-6)
    length = x.shape[-1]
    feats = []
    for k in range(max_lag + 1):
        a = x[..., : length - k]  # source, earlier
        b = x[..., k:]  # target, later
        feats.append(torch.einsum("nsl,ndl->nsd", a, b) / (length - k))
    return torch.stack(feats, dim=-1)


def lag_stack(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    """(N, C, L) -> (N, C, K, L - K) with ``out[..., k-1, t] = x[..., t + K - k]`` (lag k)."""
    length = x.shape[-1]
    return torch.stack([x[..., max_lag - k : length - k] for k in range(1, max_lag + 1)], dim=2)


@torch.no_grad()
def var_evidence(x: torch.Tensor, order: int, ridge: float = 1e-2) -> torch.Tensor:
    """Conditional (multivariate) Granger evidence from a batched ridge VAR fit.

    ``x``: (N, C, L). Returns (N, C, C, order + 1) in [src, dst]: ``log1p(Wald)`` of the lag block
    followed by the ``order`` lag coefficients ``B[k, dst, src]``. No gradient flows through it:
    it is input evidence, the network learns how much to trust it.
    """
    n, c, length = x.shape
    x = x.double()
    xl = lag_stack(x, order)  # (N, C, p, T): lag k of each target sample
    t = xl.shape[-1]
    design = xl.permute(0, 3, 2, 1).reshape(n, t, order * c)  # column index = k * C + src
    y = x[..., order:].transpose(1, 2)  # (N, T, C)
    g = design.transpose(1, 2) @ design
    lam = ridge * torch.diagonal(g, dim1=1, dim2=2).mean(dim=1)[:, None, None] + 1e-6
    g_inv = torch.linalg.inv(g + lam * torch.eye(order * c, device=x.device, dtype=x.dtype))
    w = g_inv @ design.transpose(1, 2) @ y  # (N, p*C, dst)
    sigma2 = (y - design @ w).var(dim=1) + 1e-9  # (N, dst)
    b = w.reshape(n, order, c, c)  # (N, k, src, dst)
    idx = torch.arange(order, device=x.device)[:, None] * c + torch.arange(c, device=x.device)[None]  # (p, src)
    blocks = g_inv[:, idx.T[:, :, None], idx.T[:, None, :]]  # (N, src, p, p)
    cov_inv = torch.linalg.inv(blocks)
    beta = b.permute(0, 2, 3, 1)  # (N, src, dst, k)
    wald = torch.einsum("nsdk,nskl,nsdl->nsd", beta, cov_inv, beta) / sigma2[:, None, :]
    feats = torch.cat([torch.log1p(wald)[..., None], beta * 5.0], dim=-1)
    return feats.float()


class DirectedEdgeInference(nn.Module):
    """Infer a directed adjacency per window and the per-edge lag filters used to forecast with it."""

    def __init__(self, hidden: int, max_lag: int = 8, edge_hidden: int = 64, dropout: float = 0.0, var_order: int = 5):
        super().__init__()
        self.max_lag = max_lag
        self.var_order = var_order
        self.q = nn.Linear(hidden, edge_hidden)
        self.k = nn.Linear(hidden, edge_hidden)
        n_x = 2 * (max_lag + 1) + (2 * (var_order + 1) if var_order else 0)
        self.mlp = nn.Sequential(
            nn.Linear(edge_hidden + n_x, edge_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden, edge_hidden),
            nn.GELU(),
        )
        self.gate = nn.Linear(edge_hidden, 1)
        self.lag_w = nn.Linear(edge_hidden, max_lag)
        self.self_ar = nn.Linear(hidden, max_lag)
        nn.init.constant_(self.gate.bias, -2.0)  # start sparse: sigmoid(-2) ~ 0.12

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``h``: (N, C, H) node embeddings, ``x``: (N, C, L) raw windows.

        Returns ``adj`` (N, C, C) in [src, dst] with a zero diagonal, ``logits``, the per-edge lag
        filters ``w`` (N, C, C, K) and per-node self-AR coefficients ``a`` (N, C, K).
        """
        n, c, _ = h.shape
        xc = lagged_xcorr(x, self.max_lag)  # (N, s, d, K+1)
        xc_rev = xc.transpose(1, 2)  # evidence for d -> s
        pair = self.q(h)[:, :, None, :] + self.k(h)[:, None, :, :]  # (N, s, d, E)
        feats = [pair, xc, xc_rev]
        if self.var_order:
            ve = var_evidence(x, self.var_order)
            feats += [ve, ve.transpose(1, 2)]
        e = self.mlp(torch.cat(feats, dim=-1))
        logits = self.gate(e).squeeze(-1)
        eye = torch.eye(c, device=h.device, dtype=torch.bool)
        logits = logits.masked_fill(eye, -1e4)
        adj = torch.sigmoid(logits)
        w = torch.tanh(self.lag_w(e))  # bounded so gate magnitude carries the strength
        a = self.self_ar(h)
        return {"adj": adj, "logits": logits, "w": w, "a": a}

    def forecast(self, x: torch.Tensor, out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step-ahead forecast of every sample inside the window.

        Returns ``(pred, target)``, both (N, C, L - K).
        """
        k = self.max_lag
        xl = lag_stack(x, k)  # (N, C, K, T)
        self_term = torch.einsum("nck,nckt->nct", out["a"], xl)
        cross = torch.einsum("nsd,nsdk,nskt->ndt", out["adj"], out["w"], xl)
        return self_term + cross, x[..., k:]
