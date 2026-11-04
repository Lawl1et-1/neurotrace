"""PhysioNet EEG Motor Movement/Imagery (EEG-MMI) loader, via MNE's ``eegbci`` fetcher.

64-channel referential EEG at 160 Hz. We keep the 19 standard 10-20 electrodes. Labels mark motor
execution/imagery blocks (T1/T2 annotations) rather than pathology, which makes this dataset useful
for checking that the learned graph changes with a known, benign task.
"""

from __future__ import annotations

import numpy as np

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.features import band_power
from neurotrace.data.preprocess import PreprocessConfig, filter_raw
from neurotrace.data.windows import make_windows, normalize_recording, window_labels
from neurotrace.graph.montage import STANDARD_1020


def load_eegmmi(
    subjects: list[int],
    runs: tuple[int, ...] = (4, 8, 12),  # motor imagery, left vs right fist
    cfg: PreprocessConfig | None = None,
    window_s: float = 2.0,
    stride_s: float = 1.0,
) -> list[WindowedRecording]:
    """Download (if needed) and window EEG-MMI runs; label = inside a T1/T2 task block."""
    import mne
    from mne.datasets import eegbci

    cfg = cfg or PreprocessConfig(sfreq=128.0, notch=60.0)
    recs = []
    for subj in subjects:
        for run in runs:
            (path,) = eegbci.load_data(subj, [run], verbose="ERROR")
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raw.pick(STANDARD_1020)
            raw = filter_raw(raw, cfg)
            sf = float(raw.info["sfreq"])
            data = normalize_recording(raw.get_data().astype(np.float32))
            mask = np.zeros(data.shape[1], dtype=bool)
            for a in raw.annotations:
                if a["description"] in ("T1", "T2"):
                    mask[int(a["onset"] * sf) : int((a["onset"] + a["duration"]) * sf)] = True
            windows, times = make_windows(data, sf, window_s, stride_s)
            recs.append(
                WindowedRecording(
                    subject=f"S{subj:03d}",
                    record=f"S{subj:03d}R{run:02d}",
                    sfreq=sf,
                    ch_names=list(STANDARD_1020),
                    windows=windows,
                    bands=band_power(windows, sf),
                    times=times,
                    labels=window_labels(mask, sf, window_s, stride_s),
                    window_s=window_s,
                    stride_s=stride_s,
                )
            )
    return recs
