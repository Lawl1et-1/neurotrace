"""Events, graph changes and propagation paths.

A *propagation path* is the ordered list of channels in which an event's change shows up, with a
timestamp for each. Two estimates are computed and kept side by side, because they answer
different questions:

* **graph path** - from the change of the learned directed graph during the event relative to the
  minute before it: start at the channel whose outgoing dependency grew most, then follow the
  strongest increased edges (greedy best-first). Timestamps are the first window in which each
  channel's incoming increase crosses half its peak.
* **signal path** - from band power alone: the first time each channel's power deviates from its
  pre-event baseline by more than ``z_thr`` robust standard deviations.

Both live at window resolution (the stride, 1 s by default). Synthetic couplings act at 15-40 ms
lags, so the *order* can be compared with ground truth but sub-second timing cannot be resolved.
And as everywhere, an edge here is a directed statistical dependency, not a demonstrated pathway.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotrace.eval.metrics import events_from_labels


@dataclass
class PathStep:
    channel: int
    name: str
    time_s: float
    via: int | None = None  # source channel of the edge that reached it (graph path)
    strength: float = 0.0


@dataclass
class EventExplanation:
    start: int  # window index of the detected onset
    end: int
    onset_s: float
    peak_score: float
    graph_path: list[PathStep] = field(default_factory=list)
    signal_path: list[PathStep] = field(default_factory=list)
    top_edges: list[tuple[int, int, float]] = field(default_factory=list)  # (src, dst, delta)
    delta_adj: np.ndarray | None = None
    shared_share: float = 0.0  # share of the graph increase on edges between derivations sharing an electrode
    hub: int | None = None  # channel involved in the largest share of the (unmasked) graph increase
    hub_share: float = 0.0
    hub_warning: bool = False


# A channel involved in more than this share of an event's graph increase is flagged: with C channels
# an even spread gives each about 1/C (0.06 for 18), so 0.25 is a single channel dominating, which
# points to an electrode problem, an artefact on one derivation or a reference effect, not a spread.
HUB_WARN_SHARE = 0.25


def hub_stats(inc: np.ndarray) -> tuple[int | None, float]:
    """Channel with the largest share of the total increase on its in- plus out-edges (shares sum to 1)."""
    tot = float(inc.sum())
    if tot <= 0:
        return None, 0.0
    involve = (inc.sum(axis=0) + inc.sum(axis=1)) / (2 * tot)
    h = int(np.argmax(involve))
    return h, float(involve[h])


def detect_events(score: np.ndarray, threshold: float, min_len: int = 2, merge_gap: int = 5) -> list[tuple[int, int]]:
    """Runs of ``score > threshold`` (window indices, end exclusive), merging gaps < ``merge_gap``."""
    runs = events_from_labels(np.nan_to_num(score, nan=-np.inf) > threshold)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < merge_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_len]


def delta_graph(adj: np.ndarray, start: int, end: int, baseline_s: int = 60, stride_s: float = 1.0) -> np.ndarray:
    """Mean learned adjacency during [start, end) minus the mean over the preceding baseline."""
    nb = max(1, int(baseline_s / stride_s))
    b0 = max(0, start - nb)
    base = adj[b0:start].mean(axis=0) if start > b0 else adj[:1].mean(axis=0)
    return adj[start:end].astype(np.float32).mean(axis=0) - base.astype(np.float32)


def power_deviation(bands: np.ndarray, start: int, end: int, baseline_s: int = 60, stride_s: float = 1.0) -> np.ndarray:
    """Per-channel robust z of mean log band power during [start, end) vs the preceding minute."""
    p = bands.mean(axis=-1)
    nb = max(5, int(baseline_s / stride_s))
    base = p[max(0, start - nb) : max(1, start - 5)]
    iqr = np.subtract(*np.percentile(base, [75, 25], axis=0)) + 1e-6
    return (p[start:end].mean(axis=0) - np.median(base, axis=0)) / iqr


def choose_seed(inc: np.ndarray, rule: str = "net", power_z: np.ndarray | None = None) -> int:
    """Pick the channel a path starts from.

    * ``net``: largest (outgoing - incoming) increase. Works for unbounded estimates such as the
      Granger Wald statistic, whose increase scales with coupling strength, so the seed's strong new
      outgoing edges dominate.
    * ``hybrid``: band-power deviation x (normalised outgoing increase + 0.2). The learned
      gates are bounded sigmoids: every new coupling just switches "on", so outgoing *counts* rather
      than strengths decide ``net`` and it favours nodes with many children. The seed is also where
      power rises most, which the band-power factor adds back. Chosen on v1 data, it did not hold up
      on the harder v2 benchmark (seed hit 0.22 vs 0.32 for ``net``), so ``net`` is the default (D14).
    """
    out, inn = inc.sum(axis=1), inc.sum(axis=0)
    if rule == "hybrid" and power_z is not None:
        return int(np.argmax(power_z * (out / (out.max() + 1e-12) + 0.2)))
    return int(np.argmax(out - inn))


def graph_path(adj: np.ndarray, start: int, end: int, times: np.ndarray, ch_names: list[str],
               baseline_s: int = 60, stride_s: float = 1.0, max_steps: int | None = None,
               min_rel: float = 0.25, seed_rule: str = "net",
               bands: np.ndarray | None = None, exclude: np.ndarray | None = None) -> tuple[list[PathStep], np.ndarray]:
    """Greedy best-first traversal of the increased edges of the event graph (see module doc).

    Works with any (N, C, C) directed adjacency sequence: the learned graph or a classical
    estimate such as per-window Granger. Edges below ``min_rel`` x the largest increase are ignored.
    The start channel comes from :func:`choose_seed` (``hybrid`` needs ``bands``). Edges where
    ``exclude`` is True (e.g. bipolar derivations sharing an electrode) are never walked.
    """
    d = delta_graph(adj, start, end, baseline_s, stride_s)
    inc = np.clip(d, 0, None)
    np.fill_diagonal(inc, 0)
    if exclude is not None:
        inc[exclude] = 0.0
    c = len(ch_names)
    min_delta = min_rel * float(inc.max()) if inc.max() > 0 else np.inf
    pz = power_deviation(bands, start, end, baseline_s, stride_s) if bands is not None else None
    seed = choose_seed(inc, seed_rule if pz is not None else "net", pz)
    # first-crossing time of each channel's incoming increase (and outgoing for the seed)
    nb = max(1, int(baseline_s / stride_s))
    base = adj[max(0, start - nb) : start].astype(np.float32).mean(axis=0) if start > 0 else adj[0].astype(np.float32)
    seg = adj[start:end].astype(np.float32) - base[None]

    def first_time(sig: np.ndarray) -> float:
        peak = sig.max()
        k = int(np.argmax(sig >= 0.5 * peak)) if peak > 0 else 0
        return float(times[start + k])

    steps = [PathStep(seed, ch_names[seed], first_time(np.clip(seg[:, seed, :], 0, None).sum(axis=1)))]
    visited = {seed}
    frontier = {(seed, j): inc[seed, j] for j in range(c) if j != seed}
    max_steps = max_steps or c
    while frontier and len(steps) < max_steps:
        (s, j), w = max(frontier.items(), key=lambda kv: kv[1])
        if w < min_delta:
            break
        frontier = {k: v for k, v in frontier.items() if k[1] != j}
        visited.add(j)
        t = first_time(np.clip(seg[:, :, j], 0, None).sum(axis=1))
        steps.append(PathStep(j, ch_names[j], t, via=s, strength=float(w)))
        for k in range(c):
            if k not in visited:
                frontier[(j, k)] = max(frontier.get((j, k), 0.0), inc[j, k])
    return steps, d


def signal_path(bands: np.ndarray, start: int, end: int, times: np.ndarray, ch_names: list[str],
                baseline_s: int = 60, stride_s: float = 1.0, z_thr: float = 3.0, pre: int = 5) -> list[PathStep]:
    """Order channels by when their total log band power first leaves the pre-event baseline."""
    p = bands.mean(axis=-1)  # (N, C) mean log power over bands
    nb = max(5, int(baseline_s / stride_s))
    b0 = max(0, start - pre - nb)
    base = p[b0 : max(b0 + 1, start - pre)]
    med = np.median(base, axis=0)
    iqr = np.subtract(*np.percentile(base, [75, 25], axis=0)) / 1.349 + 1e-6
    lo = max(0, start - pre)
    z = (p[lo:end] - med) / iqr
    steps = []
    for ch in range(p.shape[1]):
        hit = np.flatnonzero(z[:, ch] > z_thr)
        if len(hit):
            steps.append(PathStep(ch, ch_names[ch], float(times[lo + hit[0]]), strength=float(z[:, ch].max())))
    steps.sort(key=lambda s: (s.time_s, -s.strength))
    return steps


def explain_event(adj: np.ndarray | None, bands: np.ndarray, score: np.ndarray, start: int, end: int,
                  times: np.ndarray, ch_names: list[str], stride_s: float = 1.0, n_top: int = 10,
                  exclude: np.ndarray | None = None) -> EventExplanation:
    """Bundle graph path, signal path and top changed edges for one detected event.

    Paths always start from the ``net`` rule (largest net outgoing increase, DECISIONS D14).
    ``exclude`` (C, C) removes edges from the path and the top-edge list; how much of the increase
    sat on them is kept in ``shared_share``. The hub flag uses the increase on the kept edges.
    """
    ex = EventExplanation(start, end, float(times[start]), float(np.nanmax(score[start:end])))
    if adj is not None:
        ex.graph_path, d = graph_path(adj, start, end, times, ch_names, stride_s=stride_s, seed_rule="net",
                                      exclude=exclude)
        ex.delta_adj = d
        d = d.copy()
        inc = np.clip(d, 0, None)
        np.fill_diagonal(inc, 0)
        if exclude is not None:
            ex.shared_share = float(inc[exclude].sum() / inc.sum()) if inc.sum() > 0 else 0.0
            inc = np.where(exclude, 0.0, inc)
            d = np.where(exclude, -np.inf, d)
        ex.hub, ex.hub_share = hub_stats(inc)
        ex.hub_warning = ex.hub_share > HUB_WARN_SHARE
        np.fill_diagonal(d, -np.inf)
        flat = np.argsort(d, axis=None)[::-1][:n_top]
        ex.top_edges = [(int(i // d.shape[1]), int(i % d.shape[1]), float(d.flat[i])) for i in flat]
    ex.signal_path = signal_path(bands, start, end, times, ch_names, stride_s=stride_s)
    return ex


def path_agreement(pred: list[int], true: list[int]) -> dict[str, float]:
    """Compare a predicted channel order with the true one.

    ``seed_hit``: first predicted channel is the true seed. ``kendall``: Kendall's tau between the
    two orders over the channels present in both. ``recall``: share of true channels found.
    """
    from scipy.stats import kendalltau

    common = [c for c in pred if c in true]
    out = {
        "seed_hit": float(bool(pred) and pred[0] == true[0]),
        "recall": len(common) / len(true) if true else float("nan"),
        "kendall": float("nan"),
    }
    if len(common) >= 3:
        tau = kendalltau([pred.index(c) for c in common], [true.index(c) for c in common]).statistic
        out["kendall"] = float(tau)
    return out
