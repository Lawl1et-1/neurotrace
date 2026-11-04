"""Running a trained model over whole recordings: per-window anomaly scores and graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from neurotrace.baselines.connectivity import baseline_deviation
from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.models.neurotrace import NeuroTrace


@dataclass
class RecordingOutput:
    """Per-window outputs of the model on one recording (length N = number of windows)."""

    subject: str
    record: str
    times: np.ndarray
    labels: np.ndarray
    pred_err: np.ndarray  # next-window band-power error (NaN for the first seq_len - 1 windows)
    graph_change: np.ndarray  # deviation of the learned graph from the preceding minute (0 without one)
    var_err: np.ndarray  # within-window forecast error of the learned graph (0 without one)
    adj: np.ndarray | None  # (N, C, C) learned adjacency, float16
    prob: np.ndarray | None = None  # supervised seizure probability, if trained with a head
    extra: dict = field(default_factory=dict)

    def components(self) -> dict[str, np.ndarray]:
        return {"pred_err": self.pred_err, "graph_change": self.graph_change, "var_err": self.var_err}


@torch.no_grad()
def run_recording(model: NeuroTrace, rec: WindowedRecording, device: str = "cpu", chunk: int = 256) -> RecordingOutput:
    """Score every window of ``rec``.

    Window t is scored with the sequence of the ``seq_len`` windows ending at t: the model sees
    windows t-S+1 .. t-1 and predicts window t (causal attention, so nothing after t is used).
    """
    model.eval().to(device)
    s = model.cfg.seq_len
    n = len(rec)
    zs, adjs, var = [], [], []
    for i in range(0, n, chunk):
        x = torch.as_tensor(np.asarray(rec.windows[i : i + chunk], dtype=np.float32), device=device)
        b = model.norm_bands(torch.as_tensor(rec.bands[i : i + chunk], device=device))
        enc = model.encode_windows(x, b)
        zs.append(enc["z"])
        if "adj" in enc:
            adjs.append(enc["adj"].half().cpu())
            var.append(enc["var_err"].cpu())
    z = torch.cat(zs)
    bands_n = model.norm_bands(torch.as_tensor(rec.bands, device=device))

    pred_err = np.full(n, np.nan, dtype=np.float32)
    prob = np.full(n, np.nan, dtype=np.float32) if model.cls is not None else None
    if n >= s:
        seqs = z.unfold(0, s, 1).permute(0, 2, 1)  # (n - s + 1, S, D), sequence j ends at window j + s - 1
        for i in range(0, len(seqs), chunk):
            out = model.temporal_forward(seqs[i : i + chunk])
            end = torch.arange(i, min(i + chunk, len(seqs)), device=device) + s - 1
            if model.cfg.objective == "next":
                p = out["pred"][:, -2]  # prediction of the last window from the ones before it
            else:  # masked objective: mask the last window and reconstruct it
                m = torch.zeros(out["pred"].shape[:2], dtype=torch.bool, device=device)
                m[:, -1] = True
                p = model.temporal_forward(seqs[i : i + chunk], m)["pred"][:, -1]
            pred_err[end.cpu().numpy()] = ((p - bands_n[end]) ** 2).mean(dim=(1, 2)).cpu().numpy()
            if prob is not None:
                prob[end.cpu().numpy()] = torch.sigmoid(out["logits"][:, -1]).cpu().numpy()

    if adjs:
        adj = torch.cat(adjs).float().numpy()
        gc = baseline_deviation(adj).astype(np.float32)
        var_err = torch.cat(var).numpy()
        adj_out = adj.astype(np.float16)
    else:
        gc = np.zeros(n, np.float32)
        var_err = np.zeros(n, np.float32)
        adj_out = None
    return RecordingOutput(rec.subject, rec.record, rec.times, rec.labels, pred_err, gc, var_err, adj_out, prob)


@dataclass
class ScoreCalibration:
    """Robust (median / IQR) statistics of each score component on held-out *normal* windows."""

    stats: dict[str, tuple[float, float]]
    lambda_graph: float = 1.0
    lambda_var: float = 0.5
    smooth: int = 5
    adaptive: int = 0  # >0: z-score against the recording's own trailing `adaptive` windows (causal)

    @classmethod
    def fit(cls, outs: list[RecordingOutput], lambda_graph: float = 1.0, lambda_var: float = 0.5, smooth: int = 5,
            adaptive: int = 0) -> ScoreCalibration:
        stats = {}
        for k in ("pred_err", "graph_change", "var_err"):
            v = np.concatenate([o.components()[k][~o.labels] for o in outs])
            v = v[np.isfinite(v)]
            med = float(np.median(v)) if len(v) else 0.0
            iqr = float(np.subtract(*np.percentile(v, [75, 25]))) if len(v) else 1.0
            stats[k] = (med, iqr if iqr > 1e-8 else 1.0)
        return cls(stats, lambda_graph, lambda_var, smooth, adaptive)

    def z(self, out: RecordingOutput, key: str) -> np.ndarray:
        """Robust z-score of one component.

        With ``adaptive`` set, each window is compared with the median / IQR of the same recording's
        preceding ``adaptive`` windows (the current window excluded), so the score adapts to each
        patient's own baseline without looking ahead. Error levels differ several-fold between
        CHB-MIT patients, and a threshold fixed on one patient's scale does not transfer (D12).
        """
        v = out.components()[key].astype(np.float64)
        if self.adaptive > 0:
            import pandas as pd

            past = pd.Series(v).shift(1).rolling(self.adaptive, min_periods=30)
            med = past.median().to_numpy()
            iqr = (past.quantile(0.75) - past.quantile(0.25)).to_numpy()
            iqr = np.where(np.isfinite(iqr) & (iqr > 1e-8), iqr, np.nan)
            return np.nan_to_num((v - med) / iqr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        med, iqr = self.stats[key]
        return np.nan_to_num((v - med) / iqr, nan=0.0).astype(np.float32)

    def score(self, out: RecordingOutput) -> np.ndarray:
        """Anomaly score = z(pred_err) + lambda_graph * z(graph_change) + lambda_var * z(var_err),
        smoothed with a causal moving average of ``smooth`` windows (online-compatible)."""
        s = self.z(out, "pred_err") + self.lambda_graph * self.z(out, "graph_change") + self.lambda_var * self.z(out, "var_err")
        return causal_smooth(s, self.smooth)


def causal_smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Trailing moving average (uses only past and present values)."""
    if k <= 1:
        return x.astype(np.float32)
    c = np.cumsum(np.concatenate([np.zeros(k), x]))
    out = (c[k:] - c[:-k]) / k
    return out.astype(np.float32)
