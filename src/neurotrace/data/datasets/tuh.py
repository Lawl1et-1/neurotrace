"""TUH EEG Seizure Corpus (TUSZ) loader — pluggable, data not included.

TUSZ requires registration at https://isip.piconepress.com/projects/nedc/html/tuh_eeg/ and is
downloaded with the credentials you receive (rsync). Point ``root`` at the extracted ``edf/`` tree.
Annotations are ``.csv_bi`` files (one per EDF) with ``start_time,stop_time,label`` rows.

TUSZ recordings are referential (``EEG FP1-REF`` / ``EEG FP1-LE``), so we re-reference them to the
same 18 bipolar derivations as CHB-MIT. That lets one model and one graph be used across datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.features import band_power
from neurotrace.data.preprocess import PreprocessConfig, filter_raw
from neurotrace.data.windows import make_windows, normalize_recording, window_labels
from neurotrace.graph.montage import CHBMIT_BIPOLAR


def read_csv_bi(path: Path) -> list[tuple[float, float]]:
    """Seizure intervals from a TUSZ ``.csv_bi`` annotation file."""
    rows = [line for line in path.read_text().splitlines() if line and not line.startswith("#")]
    out = []
    for r in csv.DictReader(rows):
        if r.get("label", "").lower() == "seiz":
            out.append((float(r["start_time"]), float(r["stop_time"])))
    return out


def _to_bipolar(raw) -> np.ndarray:
    """Build the 18 CHB-MIT bipolar channels from a TUH referential recording."""
    names = {}
    for ch in raw.ch_names:
        key = ch.upper().replace("EEG ", "").split("-")[0].strip()
        names.setdefault(key, raw.ch_names.index(ch))
    data = raw.get_data()
    alias = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    out = []
    for bip in CHBMIT_BIPOLAR:
        a, b = bip.split("-")
        ia = names.get(a, names.get(alias.get(a, a)))
        ib = names.get(b, names.get(alias.get(b, b)))
        if ia is None or ib is None:
            raise KeyError(f"cannot build {bip}")
        out.append(data[ia] - data[ib])
    return np.stack(out).astype(np.float32)


def load_tuh(
    root: str | Path,
    cfg: PreprocessConfig,
    window_s: float = 2.0,
    stride_s: float = 1.0,
    max_files: int | None = None,
) -> list[WindowedRecording]:
    """Load TUSZ EDF files under ``root``. The subject id is the patient folder name."""
    import mne

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{root} not found. TUSZ needs registration; see the docstring of {__name__}."
        )
    recs = []
    for edf in sorted(root.rglob("*.edf"))[:max_files]:
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        raw = filter_raw(raw, cfg)
        try:
            data = normalize_recording(_to_bipolar(raw))
        except KeyError:
            continue
        sf = float(raw.info["sfreq"])
        mask = np.zeros(data.shape[1], dtype=bool)
        ann = edf.with_suffix(".csv_bi")
        for s, e in read_csv_bi(ann) if ann.exists() else []:
            mask[int(s * sf) : int(e * sf)] = True
        windows, times = make_windows(data, sf, window_s, stride_s)
        recs.append(
            WindowedRecording(
                subject=edf.parts[-3] if len(edf.parts) >= 3 else edf.stem,
                record=edf.stem,
                sfreq=sf,
                ch_names=list(CHBMIT_BIPOLAR),
                windows=windows,
                bands=band_power(windows, sf),
                times=times,
                labels=window_labels(mask, sf, window_s, stride_s),
                window_s=window_s,
                stride_s=stride_s,
            )
        )
    return recs
