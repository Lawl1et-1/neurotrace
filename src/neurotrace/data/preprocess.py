"""Loading and filtering EEG with MNE-Python."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PreprocessConfig:
    """Filtering/resampling parameters (defaults match ``configs/data/chbmit.yaml``)."""

    sfreq: float = 128.0
    l_freq: float | None = 0.5
    h_freq: float | None = 50.0
    notch: float | None = 60.0


def _clean_name(name: str) -> str:
    """Normalise EDF channel labels: upper case, drop MNE duplicate suffixes like ``-0``."""
    n = name.strip().upper().replace(" ", "")
    return re.sub(r"-\d+$", "", n) if n.count("-") > 1 else n


def pick_channels(raw, wanted: list[str]):
    """Keep ``wanted`` channels (case-insensitive) in that order; raise if any is missing."""
    lookup: dict[str, str] = {}
    for ch in raw.ch_names:
        lookup.setdefault(_clean_name(ch), ch)
    missing = [w for w in wanted if w.upper() not in lookup]
    if missing:
        raise KeyError(f"missing channels {missing}")
    picks = [lookup[w.upper()] for w in wanted]
    raw = raw.copy().pick(picks)
    raw.rename_channels({old: new for old, new in zip(picks, wanted, strict=True)})
    return raw


def filter_raw(raw, cfg: PreprocessConfig):
    """Notch, band-pass (zero-phase FIR) and resample an ``mne.io.Raw`` in place."""
    if cfg.notch:
        freqs = np.arange(cfg.notch, raw.info["sfreq"] / 2, cfg.notch)
        raw.notch_filter(freqs, verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")
    if cfg.sfreq and abs(raw.info["sfreq"] - cfg.sfreq) > 1e-6:
        raw.resample(cfg.sfreq, verbose="ERROR")
    return raw


def load_edf(path: str | Path, channels: list[str], cfg: PreprocessConfig) -> tuple[np.ndarray, float]:
    """Load one EDF file, keep ``channels`` and filter it.

    Returns ``(data, sfreq)`` with ``data`` of shape (C, T) in volts, float32.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raw = pick_channels(raw, channels)
    raw = filter_raw(raw, cfg)
    return raw.get_data().astype(np.float32), float(raw.info["sfreq"])


def filter_array(data: np.ndarray, sfreq: float, ch_names: list[str], cfg: PreprocessConfig):
    """Apply the same filtering to an in-memory (C, T) array (used for synthetic data in tests)."""
    import mne

    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.astype(np.float64), info, verbose="ERROR")
    raw = filter_raw(raw, cfg)
    return raw.get_data().astype(np.float32), float(raw.info["sfreq"])
