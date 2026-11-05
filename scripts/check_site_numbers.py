"""Check that every number in a piece of rendered site text comes from the results CSVs.

    python scripts/check_site_numbers.py page.txt chbmit

``page.txt`` is the visible text of the results section (e.g. ``innerText`` of ``#results``). Every
number in it must equal some value in ``results/<dataset>/*.csv`` (or a difference of two of them)
as rounded on the page: 0-3 decimals, two significant digits, or a percentage. Patient ids
(``chb01``) and a short list of constants of the method are ignored. Prints the numbers it could not
match with their context and exits non-zero if there are any.
"""

from __future__ import annotations

import itertools
import re
import sys

import numpy as np
import pandas as pd

from neurotrace.utils import RESULTS_DIR

# constants of the setup that appear in the prose (not results)
CONSTANTS = {"0", "0.001", "0.5", "1", "2", "3", "5", "8", "10", "20", "50", "99", "99.5", "95", "100"}


def forms(v: float) -> set[str]:
    out = {f"{v:.0f}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.2g}", f"{v * 100:.0f}"}
    if abs(v) < 0.01:
        out.add(f"{v:.3f}")
    return {s.lstrip("-").replace("+", "") for s in out}


# only the tables the site quotes (per-record files would make almost any number "match")
SOURCES = ("summary", "wilcoxon", "matched_fa", "term_ablation", "term_dominance", "alarm_overlap")


def allowed(dataset: str) -> set[str]:
    vals: list[float] = []
    frames = [pd.read_csv(RESULTS_DIR / dataset / f"{n}.csv") for n in SOURCES if (RESULTS_DIR / dataset / f"{n}.csv").exists()]
    pp = RESULTS_DIR / dataset / "per_patient.csv"
    if pp.exists():  # the site quotes only the full vs no-graph per-patient gap
        d = pd.read_csv(pp)
        frames.append(d[d.variant.isin(["full", "no_graph"])])
    for d in frames:
        for c in d.columns:
            if pd.api.types.is_numeric_dtype(d[c]):
                vals += [float(x) for x in d[c].dropna().unique()]
    w = pd.read_csv(RESULTS_DIR / dataset / "wilcoxon.csv") if (RESULTS_DIR / dataset / "wilcoxon.csv").exists() else None
    if w is not None:
        vals += list((w["n_patients"] - w["a_better"]).astype(float))
    ta = RESULTS_DIR / dataset / "term_ablation.csv"
    if ta.exists():  # differences between score-term rows (e.g. full - VAR only)
        a = pd.read_csv(ta)["auroc_mean"].to_numpy(dtype=float)
        vals += [x - y for x, y in itertools.permutations(a, 2)]
    s = set(CONSTANTS)
    for v in vals:
        if np.isfinite(v):
            s |= forms(v)
    return s


def main(path: str, dataset: str) -> int:
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"chb\d+[a-z]?(_\d+)?", " ", text)  # patient and record ids
    ok = allowed(dataset)
    bad = []
    for m in re.finditer(r"\d+(?:\.\d+)?", text):
        if m.group() not in ok:
            bad.append((m.group(), text[max(0, m.start() - 50) : m.end() + 30].replace("\n", " ")))
    for v, ctx in bad:
        print(f"UNMATCHED {v!r}: ...{ctx}...")
    print(f"{len(re.findall(r'[0-9]+(?:[.][0-9]+)?', text))} numbers checked, {len(bad)} unmatched")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "chbmit"))
