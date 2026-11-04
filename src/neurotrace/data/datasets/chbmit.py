"""CHB-MIT Scalp EEG Database (PhysioNet, open access).

23 paediatric subjects with intractable seizures, 256 Hz, bipolar 10-20 montage, seizure onset and
offset annotated per file in ``chbNN-summary.txt``. https://physionet.org/content/chbmit/1.0.0/
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.features import band_power
from neurotrace.data.preprocess import PreprocessConfig, load_edf
from neurotrace.data.windows import make_windows, normalize_recording, window_labels
from neurotrace.graph.montage import CHBMIT_BIPOLAR

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"

# chb21 is the same patient as chb01, recorded 1.5 years later (PhysioNet notes). Splits are by
# patient, so its recordings carry the subject id chb01 and are always held out together.
PATIENT_OF: dict[str, str] = {"chb21": "chb01"}


@dataclass
class ChbFile:
    """One EDF file and its seizure intervals (seconds from file start)."""

    subject: str
    name: str
    seizures: list[tuple[float, float]] = field(default_factory=list)


def parse_summary(text: str, subject: str) -> list[ChbFile]:
    """Parse a ``chbNN-summary.txt`` file into a list of files with seizure intervals.

    Handles both ``Seizure Start Time:`` and ``Seizure 1 Start Time:`` spellings.
    """
    files: list[ChbFile] = []
    current: ChbFile | None = None
    start: float | None = None
    for line in text.splitlines():
        m = re.match(r"\s*File Name:\s*(\S+\.edf)", line)
        if m:
            current = ChbFile(subject=subject, name=m.group(1))
            files.append(current)
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line)
        if m and current is not None:
            start = float(m.group(1))
            continue
        m = re.match(r"\s*Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", line)
        if m and current is not None and start is not None:
            current.seizures.append((start, float(m.group(1))))
            start = None
    return files


def _fetch(url: str, dest: Path, retries: int = 5) -> Path:
    """Download ``url`` to ``dest`` with retries (PhysioNet can be slow); skip if already there."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return dest
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    return dest


def select_files(files: list[ChbFile], n_background: int | None) -> list[ChbFile]:
    """All files with seizures plus ``n_background`` seizure-free files (evenly spread).

    ``None`` keeps every file. Keeping a few seizure-free hours per subject is what makes the
    false-alarm-per-hour estimate meaningful; it is still far less than the full recording.
    """
    if n_background is None:
        return files
    sz = [f for f in files if f.seizures]
    bg = [f for f in files if not f.seizures]
    if n_background and bg:
        idx = np.linspace(0, len(bg) - 1, min(n_background, len(bg))).round().astype(int)
        sz += [bg[i] for i in sorted(set(idx))]
    return sorted(sz, key=lambda f: f.name)


def download_subject(subject: str, raw_dir: str | Path, n_background: int | None = 3) -> list[ChbFile]:
    """Download the summary and the selected EDF files of one subject."""
    raw_dir = Path(raw_dir)
    summ = _fetch(f"{BASE_URL}/{subject}/{subject}-summary.txt", raw_dir / subject / f"{subject}-summary.txt")
    files = select_files(parse_summary(summ.read_text(errors="ignore"), subject), n_background)
    for f in files:
        print(f"  {subject}/{f.name} ({len(f.seizures)} seizures)")
        _fetch(f"{BASE_URL}/{subject}/{f.name}", raw_dir / subject / f.name)
    return files


def preprocess_subject(
    subject: str,
    raw_dir: str | Path,
    cache_dir: str | Path,
    cfg: PreprocessConfig,
    window_s: float,
    stride_s: float,
    bands: dict[str, tuple[float, float]],
) -> list[Path]:
    """Filter, normalise and window every downloaded file of ``subject``; write one ``.npz`` each."""
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summ = raw_dir / subject / f"{subject}-summary.txt"
    files = {f.name: f for f in parse_summary(summ.read_text(errors="ignore"), subject)}
    out = []
    for edf in sorted((raw_dir / subject).glob("*.edf")):
        dest = cache_dir / f"{edf.stem}.npz"
        if dest.exists():
            out.append(dest)
            continue
        try:
            data, sfreq = load_edf(edf, CHBMIT_BIPOLAR, cfg)
        except KeyError as e:  # a few files use a different montage; skip them, report it
            print(f"  skip {edf.name}: {e}")
            continue
        data = normalize_recording(data)
        mask = np.zeros(data.shape[1], dtype=bool)
        for s, e in files.get(edf.name, ChbFile(subject, edf.name)).seizures:
            mask[int(s * sfreq) : int(e * sfreq)] = True
        windows, times = make_windows(data, sfreq, window_s, stride_s)
        rec = WindowedRecording(
            subject=subject,
            record=edf.stem,
            sfreq=sfreq,
            ch_names=list(CHBMIT_BIPOLAR),
            windows=windows,
            bands=band_power(windows, sfreq, bands),
            times=times,
            labels=window_labels(mask, sfreq, window_s, stride_s),
            window_s=window_s,
            stride_s=stride_s,
        )
        rec.save(dest)
        out.append(dest)
        print(f"  {edf.name}: {len(rec)} windows, {int(rec.labels.sum())} seizure windows")
    return out


def load_chbmit(cache_dir: str | Path, subjects: list[str] | None = None) -> list[WindowedRecording]:
    """Load preprocessed CHB-MIT recordings from ``cache_dir`` (one ``.npz`` per EDF file)."""
    paths = sorted(Path(cache_dir).glob("chb*.npz"))
    recs = [WindowedRecording.load(p) for p in paths]
    if subjects:
        recs = [r for r in recs if r.subject in set(subjects)]
    for r in recs:
        r.subject = PATIENT_OF.get(r.subject, r.subject)
    return recs
