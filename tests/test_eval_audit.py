"""Known-easy cases for the evaluation: score sign, window/label alignment, latency."""

import numpy as np

from neurotrace.baselines.connectivity import baseline_deviation, connectivity
from neurotrace.data.windows import make_windows, window_labels
from neurotrace.eval.metrics import event_metrics, events_from_labels, window_metrics


def _switching_var(seed: int = 0, sf: int = 128, dur_s: int = 240, on: tuple[int, int] = (150, 180)):
    """Three channels; a strong coupling 0 -> 1 (lag 3) exists only during `on` (seconds)."""
    rng = np.random.default_rng(seed)
    n = dur_s * sf
    x = np.zeros((3, n))
    e = rng.standard_normal((3, n))
    mask = np.zeros(n, dtype=bool)
    mask[on[0] * sf : on[1] * sf] = True
    for t in range(3, n):
        x[0, t] = 0.6 * x[0, t - 1] + e[0, t]
        x[1, t] = 0.5 * x[1, t - 1] + (0.9 * x[0, t - 3] if mask[t] else 0.0) + e[1, t]
        x[2, t] = 0.5 * x[2, t - 1] + e[2, t]
    return x, mask


def test_window_labels_align_with_event_onset() -> None:
    sf = 128
    mask = np.zeros(60 * sf, dtype=bool)
    mask[20 * sf : 30 * sf] = True
    lab = window_labels(mask, sf, 2.0, 1.0)
    (a, b), = events_from_labels(lab)
    # window k covers [k, k+2) s; >= 50 % inside [20, 30) means k in [19, 29] (end exclusive: 30)
    assert (a, b) == (19, 30)


def test_granger_deviation_detects_coupling_switch() -> None:
    x, mask = _switching_var()
    w, _ = make_windows(x, 128, 2.0, 1.0)
    lab = window_labels(mask, 128, 2.0, 1.0)
    g = connectivity(w, 128, "granger", order=4)
    score = baseline_deviation(g)
    m = window_metrics(score, lab)
    assert m["auroc"] > 0.9, m  # positive orientation: higher score during the event
    thr = np.percentile(score[:140], 99)
    ev = event_metrics(score, lab, thr, stride_s=1.0)
    assert ev["detected"] == 1
    assert 0.0 <= ev["median_latency_s"] <= 3.0, ev


def test_early_crossing_is_a_false_alarm_not_negative_latency() -> None:
    labels = np.zeros(3600, dtype=bool)
    labels[1000:1040] = True
    score = np.zeros(3600)
    score[990] = 5.0  # 10 s before onset: noise
    score[1002:1010] = 5.0  # real detection 2 s after onset
    m = event_metrics(score, labels, threshold=1.0, stride_s=1.0, refractory_s=60.0)
    assert m["detected"] == 1
    assert m["median_latency_s"] == 2.0
    assert m["false_alarms"] == 1


def test_crossing_in_onset_window_has_latency_minus_one() -> None:
    labels = np.zeros(600, dtype=bool)
    labels[100:120] = True
    score = np.zeros(600)
    score[99:110] = 3.0
    m = event_metrics(score, labels, threshold=1.0, stride_s=1.0)
    assert m["median_latency_s"] == -1.0 and m["false_alarms"] == 0


def test_summarise_reports_subject_level_ci() -> None:
    import pandas as pd

    from neurotrace.eval.runner import summarise

    rows = []
    for i, auc in enumerate([0.6, 0.7, 0.8, 0.9]):
        for rec in range(2):  # two recordings per subject must not count as two subjects
            rows.append({"variant": "full", "subject": f"s{i}", "record": f"s{i}_{rec}", "auroc": auc,
                         "detected": 1, "n_events": 2, "false_alarms": rec, "hours": 1.0})
    out = summarise(pd.DataFrame(rows)).iloc[0]
    assert out["auroc_n"] == 4 and out["n_subjects"] == 4
    assert abs(out["auroc_mean"] - 0.75) < 1e-9
    assert abs(out["auroc_ci95"] - 3.182446 * np.std([0.6, 0.7, 0.8, 0.9], ddof=1) / 2) < 1e-4
    assert abs(out["sensitivity_mean"] - 0.5) < 1e-9 and abs(out["fa_per_hour_mean"] - 0.5) < 1e-9
