"""Cutting continuous recordings into windows and normalising them."""

from __future__ import annotations

import numpy as np


def robust_channel_stats(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel median and IQR-based scale of a (C, T) recording.

    Robust statistics are used so a few minutes of high-amplitude seizure activity do not dominate
    the normalisation of an hour-long file.
    """
    med = np.median(data, axis=1, keepdims=True)
    q75, q25 = np.percentile(data, [75, 25], axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349  # IQR of a standard normal
    scale = np.where(scale < 1e-8, 1.0, scale)
    return med, scale


def normalize_recording(data: np.ndarray) -> np.ndarray:
    """Z-score each channel of a (C, T) recording with robust recording-level statistics.

    Normalising per recording rather than per window keeps amplitude changes between windows,
    which are part of what an anomaly looks like.
    """
    med, scale = robust_channel_stats(data)
    return ((data - med) / scale).astype(np.float32)


def window_starts(n_times: int, sfreq: float, window_s: float, stride_s: float) -> np.ndarray:
    """Sample indices at which each full window starts."""
    win = int(round(window_s * sfreq))
    step = int(round(stride_s * sfreq))
    if n_times < win:
        return np.zeros(0, dtype=int)
    return np.arange(0, n_times - win + 1, step)


def make_windows(
    data: np.ndarray, sfreq: float, window_s: float, stride_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a (C, T) array into windows.

    Returns ``windows`` of shape (N, C, L) and ``times`` (N,) with each window's start in seconds.
    Uses a strided view, then copies, so memory is only allocated once.
    """
    win = int(round(window_s * sfreq))
    starts = window_starts(data.shape[1], sfreq, window_s, stride_s)
    view = np.lib.stride_tricks.sliding_window_view(data, win, axis=1)  # (C, T-win+1, win)
    windows = np.ascontiguousarray(view[:, starts, :].transpose(1, 0, 2))
    return windows, starts / sfreq


def window_labels(
    mask: np.ndarray, sfreq: float, window_s: float, stride_s: float, min_fraction: float = 0.5
) -> np.ndarray:
    """Label a window positive when at least ``min_fraction`` of its samples are positive."""
    win = int(round(window_s * sfreq))
    starts = window_starts(len(mask), sfreq, window_s, stride_s)
    csum = np.concatenate([[0], np.cumsum(mask.astype(np.int64))])
    frac = (csum[starts + win] - csum[starts]) / win
    return frac >= min_fraction
