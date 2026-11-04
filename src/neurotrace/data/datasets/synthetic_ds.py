"""Windowed version of the synthetic cohort (see :mod:`neurotrace.data.synthetic`)."""

from __future__ import annotations

import numpy as np

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.features import band_power
from neurotrace.data.synthetic import SyntheticRecording, make_synthetic_cohort
from neurotrace.data.windows import make_windows, normalize_recording, window_labels


def window_synthetic(
    rec: SyntheticRecording,
    window_s: float,
    stride_s: float,
    bands: dict[str, tuple[float, float]] | None = None,
) -> WindowedRecording:
    """Normalise and window one synthetic recording, keeping its ground truth in ``extra``.

    ``extra["adj"]`` is the (N, C, C) ground-truth directed adjacency active at each window centre.
    """
    data = normalize_recording(rec.data)
    windows, times = make_windows(data, rec.sfreq, window_s, stride_s)
    centres = times + window_s / 2
    adj = np.stack([rec.adjacency_at(t) for t in centres])
    return WindowedRecording(
        subject=rec.subject,
        record=rec.subject,
        sfreq=rec.sfreq,
        ch_names=rec.ch_names,
        windows=windows,
        bands=band_power(windows, rec.sfreq, bands),
        times=times,
        labels=window_labels(rec.event_mask(), rec.sfreq, window_s, stride_s),
        window_s=window_s,
        stride_s=stride_s,
        extra={"adj": adj, "base_adj": rec.base_adj, "events": rec.events},
    )


def load_synthetic(cfg, seed: int = 0) -> list[WindowedRecording]:
    """Build the synthetic cohort described by a ``data`` config section."""
    cohort = make_synthetic_cohort(
        n_subjects=cfg.n_subjects,
        ch_names=list(cfg.channels),
        n_edges=cfg.n_edges,
        seed=seed,
        sfreq=cfg.sfreq,
        duration_s=cfg.duration_s,
        n_events=cfg.n_events,
        event_duration_s=cfg.event_duration_s,
        event_edges=cfg.event_edges,
        mixing=cfg.mixing,
        noise_std=cfg.noise_std,
        seed_drive=cfg.get("seed_drive", 2.0),
        seed_radius=cfg.get("seed_radius", 0.90),
        event_coupling=tuple(cfg.get("event_coupling", (0.3, 0.5))),
        rhythm_change=cfg.get("rhythm_change", 1.0),
    )
    bands = {k: tuple(v) for k, v in cfg.bands.items()}
    return [window_synthetic(r, cfg.window_s, cfg.stride_s, bands) for r in cohort]
