"""Classical connectivity estimators, computed per window.

All functions take a window ``x`` of shape (C, L) and return a (C, C) matrix in the project-wide
convention ``M[src, dst]`` (row drives column). Undirected measures are symmetric.

* :func:`pearson` - absolute zero-lag correlation (undirected).
* :func:`coherence` - mean magnitude-squared coherence in a band (undirected).
* :func:`granger` - conditional Granger causality from one multivariate VAR fit (Wald statistic).
* :func:`pdc`, :func:`dtf` - Partial Directed Coherence and Directed Transfer Function from the same
  VAR model, averaged over a band.

Like the learned edges, these are directed *statistical dependencies*: volume conduction and
unobserved common inputs create spurious edges here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pearson(x: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation between channels, diagonal set to 0."""
    c = np.corrcoef(x)
    c = np.abs(np.nan_to_num(c))
    np.fill_diagonal(c, 0.0)
    return c


def coherence(x: np.ndarray, sfreq: float, band: tuple[float, float] = (1.0, 45.0), nperseg: int = 64) -> np.ndarray:
    """Magnitude-squared coherence averaged over ``band`` (Welch, 50 % overlap, Hann window)."""
    c, n = x.shape
    step = nperseg // 2
    starts = np.arange(0, n - nperseg + 1, step)
    win = np.hanning(nperseg)
    segs = np.stack([x[:, s : s + nperseg] * win for s in starts], axis=1)  # (C, S, P)
    f = np.fft.rfft(segs, axis=-1)
    freqs = np.fft.rfftfreq(nperseg, 1 / sfreq)
    sel = (freqs >= band[0]) & (freqs <= band[1])
    f = f[..., sel]
    sxy = np.einsum("isf,jsf->ijf", f, f.conj()) / len(starts)
    p = np.real(np.einsum("iif->if", sxy))
    coh = np.abs(sxy) ** 2 / (p[:, None, :] * p[None, :, :] + 1e-12)
    out = coh.mean(axis=-1)
    np.fill_diagonal(out, 0.0)
    return out


@dataclass
class VARFit:
    """Coefficients ``B[k, dst, src]`` (lag k+1), residual variances and the design Gram inverse."""

    B: np.ndarray
    sigma2: np.ndarray
    gram_inv: np.ndarray
    order: int


def fit_var(x: np.ndarray, order: int = 5, ridge: float = 1e-2) -> VARFit:
    """Least-squares VAR(order) fit with a small ridge (windows are short: 256 samples, 19 channels)."""
    c, n = x.shape
    y = x[:, order:].T  # (T, C)
    lags = np.concatenate([x[:, order - k - 1 : n - k - 1].T for k in range(order)], axis=1)  # (T, C*p)
    g = lags.T @ lags
    g_inv = np.linalg.inv(g + ridge * np.trace(g) / len(g) * np.eye(len(g)))
    w = g_inv @ lags.T @ y  # (C*p, C)
    resid = y - lags @ w
    sigma2 = resid.var(axis=0) + 1e-12
    b = w.reshape(order, c, c).transpose(0, 2, 1)  # (p, dst, src)
    return VARFit(b, sigma2, g_inv, order)


def granger(x: np.ndarray, order: int = 5, fit: VARFit | None = None) -> np.ndarray:
    """Conditional Granger causality ``src -> dst`` as the Wald statistic of the lag block.

    For each target ``dst`` and source ``src`` this tests ``B[:, dst, src] = 0`` in the full
    multivariate model, i.e. whether ``src``'s past helps predict ``dst`` given every other channel.
    One fit per window instead of C^2 restricted fits keeps it fast enough to run on every window.
    """
    fit = fit or fit_var(x, order)
    p, c, _ = fit.B.shape
    out = np.zeros((c, c))
    g = fit.gram_inv
    for src in range(c):
        idx = np.arange(p) * c + src
        cov = g[np.ix_(idx, idx)]
        cov_inv = np.linalg.inv(cov)
        beta = fit.B[:, :, src]  # (p, dst)
        wald = np.einsum("kd,kl,ld->d", beta, cov_inv, beta) / fit.sigma2
        out[src] = wald
    np.fill_diagonal(out, 0.0)
    return out


def _abar(fit: VARFit, freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """``A(f) = I - sum_k B_k exp(-i 2 pi f k / fs)``, shape (F, C, C) in [dst, src] order."""
    p, c, _ = fit.B.shape
    k = np.arange(1, p + 1)
    e = np.exp(-2j * np.pi * np.outer(freqs, k) / sfreq)  # (F, p)
    return np.eye(c)[None] - np.einsum("fk,kds->fds", e, fit.B)


def pdc(x: np.ndarray, sfreq: float, band: tuple[float, float] = (1.0, 45.0), order: int = 5,
        fit: VARFit | None = None, n_freqs: int = 32) -> np.ndarray:
    """Partial Directed Coherence (Baccala & Sameshima 2001), band-averaged, in [src, dst]."""
    fit = fit or fit_var(x, order)
    freqs = np.linspace(band[0], band[1], n_freqs)
    a = np.abs(_abar(fit, freqs, sfreq))  # (F, dst, src)
    val = a / np.sqrt((a**2).sum(axis=1, keepdims=True))  # normalise over dst for each src column
    out = val.mean(axis=0).T
    np.fill_diagonal(out, 0.0)
    return out


def dtf(x: np.ndarray, sfreq: float, band: tuple[float, float] = (1.0, 45.0), order: int = 5,
        fit: VARFit | None = None, n_freqs: int = 32) -> np.ndarray:
    """Directed Transfer Function (Kaminski & Blinowska 1991), band-averaged, in [src, dst]."""
    fit = fit or fit_var(x, order)
    freqs = np.linspace(band[0], band[1], n_freqs)
    h = np.abs(np.linalg.inv(_abar(fit, freqs, sfreq)))  # (F, dst, src)
    val = h / np.sqrt((h**2).sum(axis=2, keepdims=True))  # normalise over src for each dst row
    out = val.mean(axis=0).T
    np.fill_diagonal(out, 0.0)
    return out


METHODS = ("pearson", "coherence", "granger", "pdc", "dtf")


def connectivity(windows: np.ndarray, sfreq: float, method: str, order: int = 5) -> np.ndarray:
    """Apply one method to every window of an (N, C, L) array -> (N, C, C)."""
    out = []
    for w in windows.astype(np.float64):
        if method == "pearson":
            out.append(pearson(w))
        elif method == "coherence":
            out.append(coherence(w, sfreq))
        elif method == "granger":
            out.append(granger(w, order))
        elif method == "pdc":
            out.append(pdc(w, sfreq, order=order))
        elif method == "dtf":
            out.append(dtf(w, sfreq, order=order))
        else:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return np.stack(out).astype(np.float32)


def graph_change(adj: np.ndarray) -> np.ndarray:
    """Anomaly baseline: L1 change of a (N, C, C) connectivity sequence between consecutive windows,
    each matrix scaled by its own mean so a global power change alone does not dominate."""
    a = adj / (adj.mean(axis=(1, 2), keepdims=True) + 1e-12)
    d = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[d[0] if len(d) else 0.0], d])
