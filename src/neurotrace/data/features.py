"""Per-window spectral features."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


def band_power(
    windows: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]] | None = None
) -> np.ndarray:
    """Log band power of each channel in each window.

    Parameters
    ----------
    windows : (N, C, L) array
    sfreq : sampling frequency in Hz
    bands : mapping name -> (low, high) Hz; defaults to delta..gamma

    Returns
    -------
    (N, C, B) float32 array of ``log10`` mean PSD in each band (Welch, 1 s segments).
    """
    bands = bands or DEFAULT_BANDS
    nper = min(windows.shape[-1], int(sfreq))
    freqs, psd = welch(windows, fs=sfreq, nperseg=nper, axis=-1)
    out = []
    for lo, hi in bands.values():
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():  # band above Nyquist: fall back to the closest bin
            sel = np.argmin(np.abs(freqs - lo)) == np.arange(len(freqs))
        out.append(psd[..., sel].mean(axis=-1))
    return np.log10(np.stack(out, axis=-1) + 1e-12).astype(np.float32)
