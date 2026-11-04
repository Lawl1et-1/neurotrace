"""Common interface for EEG datasets.

Every dataset (synthetic, CHB-MIT, TUH, EEG-MMI) is turned into a list of :class:`WindowedRecording`
objects. Models, baselines and evaluation only see this structure, so adding a dataset means writing
one loader that returns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class WindowedRecording:
    """One continuous recording, already filtered, normalised and cut into windows."""

    subject: str
    record: str
    sfreq: float
    ch_names: list[str]
    windows: np.ndarray  # (N, C, L) float32/float16
    bands: np.ndarray  # (N, C, B) float32, log band power
    times: np.ndarray  # (N,) window start, seconds from the start of the recording
    labels: np.ndarray  # (N,) bool, e.g. seizure / injected event
    window_s: float
    stride_s: float
    extra: dict = field(default_factory=dict)  # dataset-specific ground truth / metadata

    def __len__(self) -> int:
        return len(self.times)

    def save(self, path: str | Path) -> None:
        """Save to a compressed ``.npz`` (windows stored as float16 to halve disk use)."""
        np.savez_compressed(
            path,
            subject=self.subject,
            record=self.record,
            sfreq=self.sfreq,
            ch_names=np.array(self.ch_names),
            windows=self.windows.astype(np.float16),
            bands=self.bands.astype(np.float32),
            times=self.times,
            labels=self.labels,
            window_s=self.window_s,
            stride_s=self.stride_s,
        )

    @classmethod
    def load(cls, path: str | Path) -> WindowedRecording:
        z = np.load(path, allow_pickle=False)
        return cls(
            subject=str(z["subject"]),
            record=str(z["record"]),
            sfreq=float(z["sfreq"]),
            ch_names=[str(c) for c in z["ch_names"]],
            windows=z["windows"],
            bands=z["bands"],
            times=z["times"],
            labels=z["labels"].astype(bool),
            window_s=float(z["window_s"]),
            stride_s=float(z["stride_s"]),
        )


def subjects_of(recs: list[WindowedRecording]) -> list[str]:
    """Sorted unique subject ids."""
    return sorted({r.subject for r in recs})
