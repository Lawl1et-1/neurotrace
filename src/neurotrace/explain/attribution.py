"""Why did the model flag this window?  Integrated Gradients, edge masking and attention.

All three explain the *prediction-error* part of the anomaly score for the sequence of ``seq_len``
windows ending at window ``t``: how badly the model, given windows t-S+1 .. t-1, predicted the
band power of window t.

* :func:`integrated_gradients` (Captum) - attribution of that error to every input: channel x window
  x frequency band (band-power input) and channel x window (raw-signal input).
* :func:`edge_masking` - remove one learned edge at a time from the whole sequence and measure how
  much the error changes; large positive values mark edges the flag depends on.
* :func:`temporal_attention` - Transformer attention of the last position over earlier windows.

Attributions explain the *model*, not the brain. A channel with large attribution is one the model's
error is sensitive to, which may or may not be where an event starts.
"""

from __future__ import annotations

import numpy as np
import torch

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.models.neurotrace import NeuroTrace


def _sequence(rec: WindowedRecording, t: int, s: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    lo = t - s + 1
    if lo < 0:
        raise ValueError(f"window {t} has fewer than seq_len={s} windows before it")
    x = torch.as_tensor(np.asarray(rec.windows[lo : t + 1], dtype=np.float32), device=device)[None]
    b = torch.as_tensor(rec.bands[lo : t + 1], device=device)[None]
    return x, b


def _error(model: NeuroTrace, x: torch.Tensor, bands: torch.Tensor, edge_mask=None) -> torch.Tensor:
    out = model(x, bands, edge_mask=edge_mask)
    target = model.norm_bands(bands[:, -1])
    return ((out["pred"][:, -2] - target) ** 2).mean(dim=(1, 2))


IG_STEPS = (32, 64, 128, 256, 512, 1024)


def integrated_gradients(model: NeuroTrace, rec: WindowedRecording, t: int, device: str = "cpu",
                         n_steps: int | None = None, tol: float = 1e-2,
                         steps: tuple[int, ...] = IG_STEPS) -> dict:
    """IG of the prediction error at window ``t``.

    Baselines: zero signal for the raw input and the training-mean band power for the band input
    (i.e. "an average, quiet window"). Returns ``bands`` (S, C, F) and ``raw`` (S, C) attributions
    (raw summed over samples).

    The integral is approximated with more and more steps until the convergence error - the gap
    between the summed attributions and F(x) - F(baseline) - is below ``tol`` (absolute; the
    relative error is reported too). ``n_steps`` fixes the number of steps instead. ``converged``
    is False if even the largest step count does not reach ``tol``; those attributions are reported
    as unconverged rather than silently used.

    Two things make the attribution complete without changing F(x): gradients flow through the
    learned graph (``attribution_mode``; it is detached in training), and the GAT keeps the top-k
    edges chosen at ``x`` along the path, so the explained function is continuous. The attribution
    therefore explains the model with its edge selection fixed to the one of the explained window.
    """
    from captum.attr import IntegratedGradients

    model.eval().to(device)
    x, b = _sequence(rec, t, model.cfg.seq_len, device)
    ig = IntegratedGradients(lambda xx, bb: _error(model, xx, bb))
    base = (torch.zeros_like(x), model.band_mean.expand_as(b).clone())
    model.attribution_mode = True  # gradients also flow through the learned graph (detached in training)
    gat = getattr(model, "spatial", None)
    freeze = gat is not None and hasattr(gat, "frozen_src")
    if freeze:  # keep the GNN's top-k edge choice of the explained input along the whole path
        with torch.no_grad():
            model(x, b)
        gat.frozen_src = gat.last_src
    try:
        with torch.no_grad():
            gap = float(_error(model, x, b) - _error(model, *base))
        for n in (n_steps,) if n_steps else steps:
            (ax, ab), delta = ig.attribute((x, b), baselines=base, n_steps=n, internal_batch_size=32,
                                           return_convergence_delta=True)
            err = float(delta.abs().max())
            if err < tol:
                break
    finally:
        model.attribution_mode = False
        if freeze:
            gat.frozen_src = None
    return {
        "bands": ab[0].detach().cpu().numpy(),
        "raw": ax[0].sum(dim=-1).detach().cpu().numpy(),
        "delta": err,
        "rel_delta": err / abs(gap) if gap else float("nan"),
        "n_steps": n,
        "converged": err < tol,
    }


@torch.no_grad()
def edge_masking(model: NeuroTrace, rec: WindowedRecording, t: int, edges: list[tuple[int, int]],
                 device: str = "cpu") -> list[tuple[int, int, float]]:
    """Change of the prediction error at ``t`` when each learned edge is removed (whole sequence).

    Positive = removing the edge lowers the error, i.e. the flag relies on that edge.
    Only meaningful for ``graph=learned`` models.
    """
    if model.cfg.graph != "learned":
        return []
    model.eval().to(device)
    x, b = _sequence(rec, t, model.cfg.seq_len, device)
    c = x.shape[2]
    ref = float(_error(model, x, b))
    out = []
    for s, d in edges:
        m = torch.ones(c, c, device=device)
        m[s, d] = 0.0
        out.append((s, d, ref - float(_error(model, x, b, edge_mask=m))))
    return sorted(out, key=lambda e: -abs(e[2]))


@torch.no_grad()
def temporal_attention(model: NeuroTrace, rec: WindowedRecording, t: int, device: str = "cpu") -> np.ndarray | None:
    """Attention (layers, S, S), averaged over heads, for the sequence ending at ``t``."""
    if model.cfg.temporal != "transformer":
        return None
    model.eval().to(device)
    x, b = _sequence(rec, t, model.cfg.seq_len, device)
    model(x, b)
    return model.temporal.last_attention[:, 0].mean(dim=1).cpu().numpy()
