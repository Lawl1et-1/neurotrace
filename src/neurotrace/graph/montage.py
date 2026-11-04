"""Electrode positions for the 10-20 system and the CHB-MIT bipolar montage.

Nodes of the graph are electrodes (referential montages) or bipolar derivations. A bipolar channel
such as ``FP1-F7`` is placed at the midpoint of its two electrodes. This is a display/prior choice,
not a claim about where the signal comes from (see docs/DECISIONS.md, D1).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

STANDARD_1020: list[str] = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
]  # fmt: skip

# The 18 bipolar derivations present in (almost) every CHB-MIT recording ("double banana").
CHBMIT_BIPOLAR: list[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]  # fmt: skip

# Older 10-20 names used by some recordings.
_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def _canonical(name: str) -> str:
    """Map an electrode label to the MNE ``standard_1020`` spelling (``FP1`` -> ``Fp1``)."""
    n = name.strip().upper()
    n = _ALIASES.get(n, n)
    if n.startswith("FP"):
        return "Fp" + n[2:]
    if n.endswith("Z"):
        return n[:-1] + "z"
    return n


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, np.ndarray]:
    """3D positions (metres, head frame) of the standard 10-20 electrodes, from MNE's built-in montage."""
    import mne

    # "standard_1020" is deprecated from MNE 1.13 in favour of the same layout named "colin27_1020"
    name = "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"
    montage = mne.channels.make_standard_montage(name)
    return {k: np.asarray(v) for k, v in montage.get_positions()["ch_pos"].items()}


def electrode_positions(ch_names: list[str]) -> np.ndarray:
    """Return an ``(n, 3)`` array of positions for referential or bipolar channel names.

    Bipolar names (``"A-B"``) map to the midpoint of electrodes A and B.
    """
    pos = _standard_positions()
    out = []
    for name in ch_names:
        parts = name.split("-")
        pts = [pos[_canonical(p)] for p in parts]
        out.append(np.mean(pts, axis=0))
    return np.stack(out)


def unit_sphere_positions(ch_names: list[str]) -> np.ndarray:
    """Positions projected onto the unit sphere (useful for plotting and for distance priors)."""
    p = electrode_positions(ch_names)
    p = p - p.mean(axis=0, keepdims=True) * np.array([1.0, 1.0, 0.0])
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def topomap_positions(ch_names: list[str]) -> np.ndarray:
    """2D azimuthal-equidistant projection for topomaps (nose up, radius ~1 at the ears)."""
    p = unit_sphere_positions(ch_names)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))  # angle from vertex
    phi = np.arctan2(y, x)
    r = theta / (np.pi / 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


def shared_electrode_mask(ch_names: list[str]) -> np.ndarray:
    """(C, C) boolean: True where two bipolar derivations share an electrode (e.g. FP1-F7 and F7-T7).

    Such pairs are coupled by construction - the shared electrode enters both signals, with
    opposite signs along a chain - so a dependency between them says nothing about the brain.
    Referential channels (no ``-``) never share. The diagonal is False.
    """
    parts = [set(n.upper().split("-")) if "-" in n else set() for n in ch_names]
    c = len(ch_names)
    m = np.zeros((c, c), dtype=bool)
    for i in range(c):
        for j in range(c):
            if i != j and parts[i] & parts[j]:
                m[i, j] = True
    return m
