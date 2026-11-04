"""Subject-level splits. Windows from one subject never appear in two splits (DECISIONS D3)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from neurotrace.data.datasets.base import WindowedRecording, subjects_of


def leave_subjects_out(
    recs: list[WindowedRecording], n_folds: int | None = None, val_subjects: int = 1, seed: int = 0
) -> Iterator[tuple[list[str], list[str], list[str]]]:
    """Yield ``(train, val, test)`` subject lists for leave-subjects-out cross-validation.

    ``n_folds=None`` means leave-one-subject-out. The validation subjects (for early stopping and
    for calibrating anomaly thresholds) are drawn from the training pool of each fold.
    """
    subs = subjects_of(recs)
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(subs))
    n_folds = n_folds or len(subs)
    folds = [order[i::n_folds] for i in range(n_folds)]
    for test in folds:
        rest = [s for s in order if s not in test]
        val = rest[:val_subjects]
        train = rest[val_subjects:]
        yield sorted(train), sorted(val), sorted(test)


def by_subject(recs: list[WindowedRecording], subjects: list[str]) -> list[WindowedRecording]:
    """Recordings belonging to ``subjects``."""
    s = set(subjects)
    return [r for r in recs if r.subject in s]
