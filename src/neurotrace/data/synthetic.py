"""Synthetic EEG with known, directed, time-varying couplings.

Ground truth for checking that a model recovers *directed statistical dependencies* before it is
trusted on real data.

Generative model (per subject)
------------------------------
* Each of the C channels is a damped oscillator: an AR(2) process with a resonance drawn from the
  theta/alpha/beta range, driven by white noise.
* A sparse baseline network of directed, lagged couplings ``src -> dst`` (lag 2-5 samples, i.e.
  15-40 ms at 128 Hz) links mostly neighbouring electrodes.
* During an *event* a seed channel switches to a strong, slow (3-7 Hz) rhythm and a directed tree
  of extra couplings rooted at the seed switches on, so the burst spreads seed -> children -> grandchildren with
  increasing delay. The order in which channels join is the ground-truth propagation path.
* Optional volume conduction: the sources are mixed by ``M = I + mixing * K`` (K = normalised
  distance kernel) before sensor noise is added, so every electrode sees a bit of its neighbours.

Convention used across the whole project: ``adj[src, dst] > 0`` means *src drives dst*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurotrace.graph.montage import STANDARD_1020
from neurotrace.graph.prior import distance_prior, knn_graph

MAX_LAG = 5
# Generator constants (tuned so events raise power ~10x at the seed, ~2-3x along the tree, see tests/test_data.py)
POLE_RADIUS = (0.85, 0.93)  # damping of the intrinsic oscillators
SEED_RADIUS = 0.90  # damping of the seed's event rhythm
BASE_COUPLING = (0.25, 0.45)
EVENT_COUPLING = (0.3, 0.5)
SEED_DRIVE = 2.0  # innovation std multiplier at the seed during an event


@dataclass
class SyntheticEvent:
    """One injected coupling-change event."""

    onset_s: float
    offset_s: float
    seed: int
    edges: list[tuple[int, int]]  # (src, dst) couplings switched on during the event
    order: list[int]  # channels sorted by when the burst reaches them (seed first)
    arrival_s: list[float]  # expected arrival delay (s) of the burst at each channel in ``order``


@dataclass
class SyntheticRecording:
    """A synthetic multichannel recording plus its ground truth."""

    subject: str
    data: np.ndarray  # (C, T) float32, sensor space
    sfreq: float
    ch_names: list[str]
    base_adj: np.ndarray  # (C, C) bool, adj[src, dst]
    events: list[SyntheticEvent] = field(default_factory=list)

    @property
    def n_times(self) -> int:
        return self.data.shape[1]

    def event_mask(self) -> np.ndarray:
        """Boolean (T,) mask of samples that fall inside an event."""
        m = np.zeros(self.n_times, dtype=bool)
        for ev in self.events:
            m[int(ev.onset_s * self.sfreq) : int(ev.offset_s * self.sfreq)] = True
        return m

    def adjacency_at(self, t_s: float) -> np.ndarray:
        """Ground-truth directed adjacency (C, C) active at time ``t_s`` (seconds)."""
        a = self.base_adj.copy()
        for ev in self.events:
            if ev.onset_s <= t_s < ev.offset_s:
                for s, d in ev.edges:
                    a[s, d] = True
        return a


def _ar2_coeffs(freq: float, sfreq: float, r: float) -> tuple[float, float]:
    """AR(2) coefficients of a resonator with pole radius ``r`` at ``freq`` Hz."""
    w = 2 * np.pi * freq / sfreq
    return 2 * r * np.cos(w), -(r**2)


def _companion_radius(self_coef: np.ndarray, coup: np.ndarray) -> float:
    """Spectral radius of the VAR(MAX_LAG) companion matrix. ``coup`` is (MAX_LAG, C, C) [dst, src]."""
    c = self_coef.shape[1]
    big = np.zeros((MAX_LAG * c, MAX_LAG * c))
    for k in range(MAX_LAG):
        block = coup[k].copy()
        if k < 2:
            block[np.diag_indices(c)] += self_coef[k]
        big[:c, k * c : (k + 1) * c] = block
    big[c:, :-c] = np.eye((MAX_LAG - 1) * c)
    return float(np.max(np.abs(np.linalg.eigvals(big))))


def _random_tree(seed: int, n_edges: int, neighbours: np.ndarray, rng: np.random.Generator):
    """Grow a directed tree from ``seed`` along spatial neighbours (BFS-like, random order)."""
    edges: list[tuple[int, int]] = []
    reached = [seed]
    depth = {seed: 0}
    frontier = [seed]
    while len(edges) < n_edges and frontier:
        src = frontier.pop(0)
        cand = [j for j in np.flatnonzero(neighbours[src]) if j not in depth]
        rng.shuffle(cand)
        for dst in cand[: rng.integers(1, 3)]:
            if len(edges) >= n_edges:
                break
            edges.append((src, int(dst)))
            depth[int(dst)] = depth[src] + 1
            reached.append(int(dst))
            frontier.append(int(dst))
    return edges, reached, depth


def make_population_network(
    ch_names: list[str], n_edges: int, rng: np.random.Generator
) -> np.ndarray:
    """Sparse directed baseline network on (mostly) neighbouring electrodes. Returns adj[src, dst]."""
    c = len(ch_names)
    nb = knn_graph(ch_names, k=4)
    pairs = [(i, j) for i in range(c) for j in range(c) if nb[i, j]]
    chosen = rng.choice(len(pairs), size=n_edges, replace=False)
    adj = np.zeros((c, c), dtype=bool)
    for k in chosen:
        i, j = pairs[k]
        if not adj[j, i]:  # avoid trivial 2-cycles in the baseline
            adj[i, j] = True
    return adj


def simulate_subject(
    subject: str,
    base_adj: np.ndarray,
    ch_names: list[str] | None = None,
    sfreq: float = 128.0,
    duration_s: float = 240.0,
    n_events: int = 3,
    event_duration_s: float = 12.0,
    event_edges: int = 6,
    mixing: float = 0.15,
    noise_std: float = 1.0,
    seed: int = 0,
    seed_drive: float = SEED_DRIVE,
    seed_radius: float = SEED_RADIUS,
    event_coupling: tuple[float, float] = EVENT_COUPLING,
    rhythm_change: float = 1.0,
) -> SyntheticRecording:
    """Simulate one subject. ``base_adj`` is jittered slightly so subjects are not identical."""
    rng = np.random.default_rng(seed)
    ch_names = list(ch_names or STANDARD_1020)
    c = len(ch_names)
    n = int(duration_s * sfreq)
    nb = knn_graph(ch_names, k=4)

    # subject-specific jitter of the population network: drop one edge, add one neighbour edge
    adj = base_adj.copy()
    on = np.argwhere(adj)
    if len(on) > 2:
        i, j = on[rng.integers(len(on))]
        adj[i, j] = False
    cand = np.argwhere(nb & ~adj & ~adj.T)
    i, j = cand[rng.integers(len(cand))]
    adj[i, j] = True

    # intrinsic oscillators
    freqs = rng.uniform(5.0, 22.0, size=c)
    r = rng.uniform(*POLE_RADIUS, size=c)
    self_coef = np.array([_ar2_coeffs(f, sfreq, rr) for f, rr in zip(freqs, r, strict=True)]).T

    def coupling_tensor(a: np.ndarray, strength: tuple[float, float]) -> np.ndarray:
        coup = np.zeros((MAX_LAG, c, c))
        for s, d in np.argwhere(a):
            lag = rng.integers(2, MAX_LAG + 1)
            coup[lag - 1, d, s] = rng.uniform(*strength)
        return coup

    base_coup = coupling_tensor(adj, BASE_COUPLING)
    while _companion_radius(self_coef, base_coup) >= 0.97:
        base_coup *= 0.9

    # events: non-overlapping, away from the edges of the recording
    events: list[SyntheticEvent] = []
    event_coups: list[np.ndarray] = []
    event_self: list[np.ndarray] = []
    slots = np.linspace(0.1 * duration_s, 0.9 * duration_s - event_duration_s, max(n_events, 1))
    for k in range(n_events):
        onset = float(slots[k] + rng.uniform(-0.03, 0.03) * duration_s)
        seed_ch = int(rng.integers(c))
        edges, order, _ = _random_tree(seed_ch, event_edges, nb, rng)
        # the seed switches to a slow, sharply tuned rhythm (seizure-like) with stronger drive
        sc = self_coef.copy()
        # with probability `rhythm_change` the seed also retunes to a slow rhythm (a spectral
        # signature); otherwise the event is a coupling change plus a modest drive increase only
        if rng.random() < rhythm_change:
            sc[:, seed_ch] = _ar2_coeffs(rng.uniform(3.0, 7.0), sfreq, seed_radius)
        coup = np.zeros((MAX_LAG, c, c))
        lags = {}
        for s, d in edges:
            lag = int(rng.integers(3, MAX_LAG + 1))
            lags[(s, d)] = lag
            coup[lag - 1, d, s] = rng.uniform(*event_coupling)
        while _companion_radius(sc, base_coup + coup) >= 0.975:
            coup *= 0.9
        arrival = {seed_ch: 0.0}
        for s, d in edges:  # edges are listed in BFS order, so parents come first
            arrival[d] = arrival[s] + lags[(s, d)] / sfreq
        events.append(
            SyntheticEvent(
                onset_s=onset,
                offset_s=onset + event_duration_s,
                seed=seed_ch,
                edges=edges,
                order=sorted(order, key=lambda o: (arrival[o], order.index(o))),
                arrival_s=sorted(arrival[o] for o in order),
            )
        )
        event_coups.append(coup)
        event_self.append(sc)

    # simulate the sources
    x = np.zeros((c, n + MAX_LAG))
    eps = rng.standard_normal((c, n + MAX_LAG))
    state = np.full(n + MAX_LAG, -1, dtype=int)
    for k, ev in enumerate(events):
        state[MAX_LAG + int(ev.onset_s * sfreq) : MAX_LAG + int(ev.offset_s * sfreq)] = k
        eps[ev.seed, state == k] *= seed_drive
    for t in range(MAX_LAG, n + MAX_LAG):
        if state[t] < 0:
            coup, sc = base_coup, self_coef
        else:
            coup, sc = base_coup + event_coups[state[t]], event_self[state[t]]
        past = x[:, t - MAX_LAG : t][:, ::-1]  # (C, MAX_LAG), column k = lag k+1
        val = sc[0] * past[:, 0] + sc[1] * past[:, 1]
        val += np.einsum("kds,sk->d", coup, past)
        val += eps[:, t]
        x[:, t] = val
    x = x[:, MAX_LAG:]

    # volume conduction + sensor noise
    if mixing > 0:
        k = distance_prior(ch_names, sigma=0.35)
        k = k / k.sum(axis=1, keepdims=True)
        x = (np.eye(c) + mixing * k) @ x
    # scale by the std of event-free samples (as a clinician's baseline would)
    calm = np.ones(n, dtype=bool)
    for ev in events:
        calm[int(ev.onset_s * sfreq) : int(ev.offset_s * sfreq)] = False
    x = x / x[:, calm].std(axis=1, keepdims=True)
    x = x + 0.3 * noise_std * rng.standard_normal(x.shape)

    return SyntheticRecording(
        subject=subject,
        data=x.astype(np.float32),
        sfreq=sfreq,
        ch_names=ch_names,
        base_adj=adj,
        events=events,
    )


def make_synthetic_cohort(
    n_subjects: int = 8,
    ch_names: list[str] | None = None,
    n_edges: int = 12,
    seed: int = 0,
    **kwargs,
) -> list[SyntheticRecording]:
    """Simulate ``n_subjects`` subjects that share a population network (with per-subject jitter)."""
    ch_names = list(ch_names or STANDARD_1020)
    rng = np.random.default_rng(seed)
    pop = make_population_network(ch_names, n_edges, rng)
    return [
        simulate_subject(f"syn{s:02d}", pop, ch_names=ch_names, seed=seed * 1000 + s + 1, **kwargs)
        for s in range(n_subjects)
    ]
