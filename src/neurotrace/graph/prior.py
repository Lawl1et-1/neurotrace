"""Distance-based prior adjacency from electrode coordinates."""

from __future__ import annotations

import numpy as np

from neurotrace.graph.montage import unit_sphere_positions


def distance_prior(ch_names: list[str], sigma: float = 0.5, self_loops: bool = False) -> np.ndarray:
    """Gaussian kernel on great-circle-ish (chord) distances between electrodes.

    ``A[i, j] = exp(-d_ij^2 / sigma^2)``, symmetric, values in (0, 1]. ``sigma`` is in units of the
    head radius. The prior is undirected on purpose: distance alone says nothing about direction.
    """
    p = unit_sphere_positions(ch_names)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    a = np.exp(-(d**2) / sigma**2)
    if not self_loops:
        np.fill_diagonal(a, 0.0)
    return a.astype(np.float32)


def knn_graph(ch_names: list[str], k: int = 4) -> np.ndarray:
    """Binary symmetric k-nearest-neighbour graph (used to place plausible synthetic couplings)."""
    p = unit_sphere_positions(ch_names)
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    a = np.zeros_like(d, dtype=bool)
    rows = np.repeat(np.arange(len(ch_names)), k)
    a[rows, idx.ravel()] = True
    return a | a.T
