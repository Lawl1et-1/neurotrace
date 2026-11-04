"""Sequences of consecutive windows for training."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from neurotrace.data.datasets.base import WindowedRecording


class SequenceDataset(Dataset):
    """All runs of ``seq_len`` consecutive windows inside each recording.

    ``normal_only=True`` drops sequences that contain any labelled window, so the self-supervised
    model only learns what *normal* activity looks like (labels are never used as inputs).
    ``stride`` subsamples start positions to cut redundancy between overlapping sequences.
    """

    def __init__(
        self,
        recs: list[WindowedRecording],
        seq_len: int,
        normal_only: bool = False,
        stride: int = 1,
        with_adj: bool = False,
    ):
        self.recs = recs
        self.seq_len = seq_len
        self.with_adj = with_adj and all("adj" in r.extra for r in recs)
        index = []
        for ri, r in enumerate(recs):
            n = len(r) - seq_len + 1
            if n <= 0:
                continue
            starts = np.arange(0, n, stride)
            if normal_only:
                bad = np.convolve(r.labels.astype(int), np.ones(seq_len, dtype=int), "valid")[starts] > 0
                starts = starts[~bad]
            index += [(ri, int(s)) for s in starts]
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ri, s = self.index[i]
        r = self.recs[ri]
        sl = slice(s, s + self.seq_len)
        item = {
            "x": torch.from_numpy(np.asarray(r.windows[sl], dtype=np.float32)),
            "bands": torch.from_numpy(np.asarray(r.bands[sl], dtype=np.float32)),
            "labels": torch.from_numpy(r.labels[sl].astype(np.float32)),
        }
        if self.with_adj:
            item["adj_true"] = torch.from_numpy(r.extra["adj"][sl].astype(np.float32))
        return item


def band_stats(recs: list[WindowedRecording]) -> tuple[np.ndarray, np.ndarray]:
    """Per-(channel, band) mean and std of log band power over the given (training) recordings."""
    allb = np.concatenate([r.bands for r in recs], axis=0)
    return allb.mean(axis=0), allb.std(axis=0)
