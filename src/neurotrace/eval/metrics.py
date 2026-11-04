"""Evaluation metrics: edge recovery, window-level detection, event-level latency and false alarms."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def offdiag(a: np.ndarray) -> np.ndarray:
    """Flatten the off-diagonal entries of (..., C, C) to (..., C*(C-1))."""
    c = a.shape[-1]
    m = ~np.eye(c, dtype=bool)
    return a[..., m]


def edge_auroc(pred: np.ndarray, true: np.ndarray) -> float:
    """AUROC of predicted edge scores vs a binary directed ground truth (off-diagonal entries only).

    Works for one matrix or a stack (N, C, C): with a stack, entries of all windows are pooled.
    """
    p, t = offdiag(pred).ravel(), offdiag(true).ravel().astype(bool)
    if t.all() or not t.any():
        return float("nan")
    return float(roc_auc_score(t, p))


def direction_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """For true edges s->d whose reverse d->s is absent, fraction where pred[s, d] > pred[d, s]."""
    t = true.astype(bool)
    hits, total = 0, 0
    for s, d in np.argwhere(t & ~t.T):
        total += 1
        hits += pred[s, d] > pred[d, s]
    return hits / total if total else float("nan")


def window_metrics(score: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Window-level AUROC and AUPRC (NaN scores are dropped)."""
    ok = np.isfinite(score)
    s, y = score[ok], labels[ok].astype(bool)
    if y.all() or not y.any():
        return {"auroc": float("nan"), "auprc": float("nan"), "prevalence": float(y.mean()) if len(y) else float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s)), "prevalence": float(y.mean())}


def events_from_labels(labels: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of positive windows as (start, end_exclusive) index pairs."""
    y = np.concatenate([[0], labels.astype(int), [0]])
    d = np.diff(y)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def event_metrics(
    score: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    stride_s: float,
    refractory_s: float = 60.0,
    max_latency_s: float = 60.0,
    pre_windows: int = 1,
    post_grace_s: float = 10.0,
) -> dict[str, float]:
    """Event-level evaluation of an online detector that alarms when ``score > threshold``.

    * An event spanning windows [a, b) is *detected* when the score exceeds the threshold anywhere
      in [a - pre_windows, min(b, a + max_latency)). ``pre_windows = 1`` only admits the window that
      straddles the onset (a window is labelled positive when at least half of it is inside the
      event, so the one before can already contain the onset). Latency = first such window - a, in
      seconds; it is therefore never below ``-pre_windows * stride_s``.
    * Alarms are the rising edges of ``score > threshold``, grouped with a ``refractory_s`` dead time.
    * A false alarm is an alarm outside every [a - pre_windows, b + post_grace) interval. An early
      crossing before that interval is a false alarm, never a detection with negative latency.
    * False alarms per hour use the duration of the recording outside events.
    """
    s = np.nan_to_num(score, nan=-np.inf)
    above = s > threshold
    alarms: list[int] = []
    last = -np.inf
    for i in np.flatnonzero(above):
        if (i - last) * stride_s >= refractory_s:
            alarms.append(int(i))
            last = i
    events = events_from_labels(labels)
    lat_win = int(round(max_latency_s / stride_s))
    grace = int(round(post_grace_s / stride_s))
    detected, latencies = 0, []
    for a, b in events:
        lo, hi = max(0, a - pre_windows), min(b, a + lat_win)
        hit = np.flatnonzero(above[lo:hi])
        if len(hit):
            detected += 1
            latencies.append((lo + int(hit[0]) - a) * stride_s)
    covered = np.zeros(len(labels), dtype=bool)
    for a, b in events:
        covered[max(0, a - pre_windows) : b + grace] = True
    false_alarms = sum(1 for al in alarms if not covered[al])
    hours = (len(labels) - labels.sum()) * stride_s / 3600.0
    return {
        "n_events": len(events),
        "detected": detected,
        "sensitivity": detected / len(events) if events else float("nan"),
        "median_latency_s": float(np.median(latencies)) if latencies else float("nan"),
        "false_alarms": false_alarms,
        "fa_per_hour": false_alarms / hours if hours > 0 else float("nan"),
        "hours": hours,
    }
