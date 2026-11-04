import numpy as np
import pytest

from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.datasets.chbmit import parse_summary, select_files
from neurotrace.data.features import band_power
from neurotrace.data.splits import leave_subjects_out
from neurotrace.data.synthetic import make_synthetic_cohort
from neurotrace.data.windows import make_windows, normalize_recording, window_labels

SUMMARY = """
File Name: chb01_01.edf
Number of Seizures in File: 0

File Name: chb01_03.edf
Number of Seizures in File: 1
Seizure Start Time: 2996 seconds
Seizure End Time: 3036 seconds

File Name: chb01_04.edf
Number of Seizures in File: 2
Seizure 1 Start Time: 10 seconds
Seizure 1 End Time: 20 seconds
Seizure 2 Start Time: 100 seconds
Seizure 2 End Time: 130 seconds
"""


def test_windows_shape_and_times() -> None:
    x = np.arange(2 * 1000, dtype=np.float32).reshape(2, 1000)
    w, t = make_windows(x, sfreq=100, window_s=2.0, stride_s=1.0)
    assert w.shape == (9, 2, 200)
    np.testing.assert_allclose(t, np.arange(9.0))
    np.testing.assert_array_equal(w[3, 1], x[1, 300:500])


def test_window_labels_majority() -> None:
    mask = np.zeros(1000, dtype=bool)
    mask[250:450] = True
    lab = window_labels(mask, sfreq=100, window_s=2.0, stride_s=1.0)
    # windows 2 (200-400) and 3 (300-500) are 75 % positive; windows 1 and 4 only 25 %
    assert lab.tolist()[:5] == [False, False, True, True, False]


def test_normalize_is_robust_zscore() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(3, 20000))
    x[:, :200] += 500  # a short artefact should not change the scale much
    z = normalize_recording(x)
    assert np.allclose(np.median(z, axis=1), 0, atol=0.05)
    assert np.allclose(np.percentile(z[:, 1000:], 84, axis=1), 1.0, atol=0.1)


def test_band_power_finds_alpha() -> None:
    sf = 128
    t = np.arange(4 * sf) / sf
    sig = np.sin(2 * np.pi * 10 * t)[None, None, :]  # 10 Hz
    bp = band_power(sig, sf)
    assert bp.shape == (1, 1, 5)
    assert int(np.argmax(bp[0, 0])) == 2  # alpha


def test_bandpass_filter_attenuates_line_noise() -> None:
    from neurotrace.data.preprocess import PreprocessConfig, filter_array

    sf = 256
    t = np.arange(20 * sf) / sf
    x = np.stack([np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)] * 2)
    y, sf2 = filter_array(x, sf, ["Cz", "Pz"], PreprocessConfig(sfreq=128, notch=60.0))
    assert sf2 == 128
    bp = band_power(y[None, :, 2 * 128 : -2 * 128], sf2, {"alpha": (8, 13), "line": (58, 62)})
    assert bp[0, 0, 0] - bp[0, 0, 1] > 3  # line noise > 30 dB below alpha (and above Nyquist)


def test_parse_summary_both_spellings() -> None:
    files = parse_summary(SUMMARY, "chb01")
    assert [f.name for f in files] == ["chb01_01.edf", "chb01_03.edf", "chb01_04.edf"]
    assert files[1].seizures == [(2996.0, 3036.0)]
    assert files[2].seizures == [(10.0, 20.0), (100.0, 130.0)]
    sel = select_files(files, n_background=0)
    assert [f.name for f in sel] == ["chb01_03.edf", "chb01_04.edf"]


def _rec(subject: str) -> WindowedRecording:
    return WindowedRecording(
        subject, subject, 128.0, ["Cz"], np.zeros((2, 1, 4)), np.zeros((2, 1, 5)),
        np.arange(2.0), np.zeros(2, bool), 2.0, 1.0,
    )  # fmt: skip


def test_leave_subjects_out_has_no_leakage() -> None:
    recs = [_rec(f"s{i}") for i in range(6)] + [_rec("s0")]
    seen_test = []
    for train, val, test in leave_subjects_out(recs, val_subjects=1):
        assert not set(train) & set(test)
        assert not set(val) & set(test)
        assert not set(train) & set(val)
        seen_test += test
    assert sorted(seen_test) == [f"s{i}" for i in range(6)]


@pytest.fixture(scope="module")
def cohort():
    return make_synthetic_cohort(n_subjects=2, duration_s=60, n_events=2, event_duration_s=8, seed=1)


def test_synthetic_ground_truth(cohort) -> None:
    rec = cohort[0]
    assert rec.data.shape == (19, 60 * 128)
    assert np.isfinite(rec.data).all()
    assert rec.base_adj.sum() >= 8
    assert not np.diag(rec.base_adj).any()
    ev = rec.events[0]
    assert ev.order[0] == ev.seed
    assert ev.arrival_s == sorted(ev.arrival_s)
    during = rec.adjacency_at((ev.onset_s + ev.offset_s) / 2)
    for s, d in ev.edges:
        assert during[s, d]
    # the burst raises power at the seed during the event
    sf = int(rec.sfreq)
    inside = rec.data[ev.seed, int(ev.onset_s * sf) + sf : int(ev.offset_s * sf)].var()
    outside = rec.data[ev.seed, : int(ev.onset_s * sf)].var()
    assert inside > 1.5 * outside


def test_synthetic_is_reproducible() -> None:
    a = make_synthetic_cohort(n_subjects=1, duration_s=20, n_events=1, event_duration_s=5, seed=3)
    b = make_synthetic_cohort(n_subjects=1, duration_s=20, n_events=1, event_duration_s=5, seed=3)
    np.testing.assert_array_equal(a[0].data, b[0].data)
