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
) -> dict[str, float]:
    """Event-level evaluation of an online detector that alarms when ``score > threshold``.

    * An event is *detected* if an alarm starts inside it or within ``max_latency_s`` after onset.
      Latency = first alarm time - onset (seconds; can be <0 if the alarm starts just before).
    * Alarms are grouped: after an alarm, new alarms are suppressed for ``refractory_s``.
    * A false alarm is an alarm that does not overlap any event (+/- the latency window).
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
    detected, latencies = 0, []
    matched = set()
    for a, b in events:
        hits = [al for al in alarms if a - lat_win // 4 <= al < min(b, a + lat_win) + 1]
        if hits:
            detected += 1
            latencies.append((hits[0] - a) * stride_s)
            matched.update(hits)
        # alarms during the event that started earlier also count as matched
        matched.update(al for al in alarms if a <= al < b + lat_win)
    false_alarms = len([al for al in alarms if al not in matched])
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
