"""Post-hoc analysis of saved per-window scores (``results/<dataset>/scores/``, written by evaluate.py).

    python scripts/analyze_results.py chbmit

Writes to ``results/<dataset>/``:

* ``per_patient.csv``      - window AUROC and AUPRC of every detector for every test patient.
* ``wilcoxon.csv``         - paired two-sided Wilcoxon signed-rank tests over patients.
* ``matched_fa.csv``       - sensitivity and latency at 0.5 / 1 / 2 false alarms per hour (per patient
                             in ``matched_fa_rows.csv``), threshold set on the test patient (oracle).
* ``term_dominance.csv``   - for each model variant: AUROC of each score term alone and each term's
                             share of the score at alarm windows.
* ``alarm_overlap.csv``    - Jaccard overlap of alarm windows between model variants.
* ``summary.csv``          - rebuilt with the combined detectors appended (window metrics only).

Combined detectors (no tuning, equal weights of causal robust z-scores, then the usual 5-window
causal smoothing): ``combined_bp_graph`` = band-power deviation + the full model's learned-graph
change; ``combined_bp_granger`` = band-power deviation + Granger connectivity change.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from neurotrace.eval.metrics import operating_points_at_fa, window_metrics
from neurotrace.eval.runner import summarise
from neurotrace.eval.scoring import causal_smooth
from neurotrace.utils import RESULTS_DIR

TARGETS = (0.5, 1.0, 2.0)
TERMS = ("pred_err", "graph_change", "var_err")
PAIRS = [("full", "no_graph"), ("full", "baseline_bandpower_dev"), ("combined_bp_graph", "baseline_bandpower_dev"),
         ("combined_bp_graph", "combined_bp_granger"), ("full", "baseline_granger")]


def load_scores(root: Path) -> dict[str, dict[str, dict]]:
    """detector -> record -> arrays and metadata."""
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in sorted(root.glob("*/*.npz")):
        z = np.load(f)
        out[f.parent.name][f.stem] = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
    return out


def add_combined(scores: dict) -> None:
    bp = scores.get("baseline_bandpower_dev", {})
    for name, other, key in (("combined_bp_graph", "full", "z_graph_change"), ("combined_bp_granger", "baseline_granger", "z_raw")):
        src = scores.get(other, {})
        for rec, d in bp.items():
            if rec in src and key in src[rec]:
                s = causal_smooth(d["z_raw"] + src[rec][key], 5)
                scores[name][rec] = {**{k: d[k] for k in ("labels", "subject", "stride_s")}, "score": s,
                                     "threshold": np.nan, "fold": src[rec]["fold"]}


def per_record_rows(scores: dict) -> pd.DataFrame:
    rows = []
    for det, recs in scores.items():
        for rec, d in recs.items():
            row = {"variant": det, "fold": d["fold"], "subject": str(d["subject"]), "record": rec}
            row.update(window_metrics(d["score"], d["labels"].astype(bool)))
            rows.append(row)
    return pd.DataFrame(rows)


def matched_fa(scores: dict) -> pd.DataFrame:
    rows = []
    for det, recs in scores.items():
        by_subj = defaultdict(list)
        for d in recs.values():
            by_subj[str(d["subject"])].append(d)
        for subj, ds in by_subj.items():
            if not any(d["labels"].any() for d in ds):
                continue
            ops = operating_points_at_fa([d["score"] for d in ds], [d["labels"].astype(bool) for d in ds],
                                         float(ds[0]["stride_s"]), TARGETS)
            for t, op in ops.items():
                rows.append({"variant": det, "subject": subj, "target_fa_h": t, **op})
        print(f"  matched FA {det}")
    return pd.DataFrame(rows)


def ci95(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def term_dominance(scores: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, alarms = [], {}
    for det, recs in scores.items():
        first = next(iter(recs.values()))
        if "z_graph_change" not in first:
            continue
        per_subj = defaultdict(lambda: defaultdict(list))
        share_num, share_den = defaultdict(float), 0.0
        al = {}
        for rec, d in recs.items():
            y = d["labels"].astype(bool)
            terms = {k: float(d[f"w_{k}"]) * d[f"z_{k}"] for k in TERMS}
            if y.any() and not y.all():
                for k, v in terms.items():
                    per_subj[str(d["subject"])][k].append(window_metrics(causal_smooth(v, int(d["smooth"])), y)["auroc"])
            hit = d["score"] > d["threshold"]
            al[rec] = hit
            if hit.any():
                sm = {k: causal_smooth(v, int(d["smooth"]))[hit] for k, v in terms.items()}
                tot = sum(np.abs(v) for v in sm.values()) + 1e-9
                for k in TERMS:
                    share_num[k] += float((np.abs(sm[k]) / tot).sum())
                share_den += float(hit.sum())
        alarms[det] = al
        row = {"variant": det, "alarm_windows": int(share_den)}
        for k in TERMS:
            v = np.array([np.nanmean(per_subj[s][k]) for s in per_subj], dtype=float)
            row[f"auroc_{k}_mean"], row[f"auroc_{k}_ci95"] = float(np.nanmean(v)), ci95(v)
            row[f"alarm_share_{k}"] = share_num[k] / share_den if share_den else float("nan")
        rows.append(row)
    over = []
    names = sorted(alarms)
    for a in names:
        for b in names:
            if a < b:
                common = set(alarms[a]) & set(alarms[b])
                inter = sum(int((alarms[a][r] & alarms[b][r]).sum()) for r in common)
                union = sum(int((alarms[a][r] | alarms[b][r]).sum()) for r in common)
                over.append({"a": a, "b": b, "jaccard_alarm_windows": inter / union if union else float("nan")})
    return pd.DataFrame(rows), pd.DataFrame(over)


def wilcoxon(per_patient: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("auroc", "auprc"):
        wide = per_patient.pivot(index="subject", columns="variant", values=metric)
        for a, b in PAIRS:
            if a not in wide or b not in wide:
                continue
            d = (wide[a] - wide[b]).dropna()
            res = stats.wilcoxon(d) if len(d) >= 2 and (d != 0).any() else None
            rows.append({"metric": metric, "a": a, "b": b, "n_patients": len(d), "median_diff": float(d.median()),
                         "a_better": int((d > 0).sum()), "W": float(res.statistic) if res else float("nan"),
                         "p_value": float(res.pvalue) if res else float("nan")})
    return pd.DataFrame(rows)


def main(dataset: str) -> None:
    out = RESULTS_DIR / dataset
    scores = load_scores(out / "scores")
    if not scores:
        raise SystemExit(f"no saved scores in {out / 'scores'}; run evaluate.py first")
    add_combined(scores)

    rec_rows = per_record_rows(scores)
    per_patient = rec_rows.groupby(["variant", "subject"])[["auroc", "auprc"]].mean().reset_index()
    per_patient.to_csv(out / "per_patient.csv", index=False)
    wilcoxon(per_patient).to_csv(out / "wilcoxon.csv", index=False)

    mf = matched_fa(scores)
    mf.to_csv(out / "matched_fa_rows.csv", index=False)
    agg = []
    for (det, t), d in mf.groupby(["variant", "target_fa_h"]):
        r = {"variant": det, "target_fa_h": t, "n_subjects": len(d)}
        for m in ("sensitivity", "median_latency_s", "fa_per_hour"):
            v = d[m].to_numpy(dtype=float)
            r[f"{m}_mean"], r[f"{m}_ci95"], r[f"{m}_n"] = float(np.nanmean(v)), ci95(v), int(np.isfinite(v).sum())
        agg.append(r)
    pd.DataFrame(agg).to_csv(out / "matched_fa.csv", index=False)

    td, ov = term_dominance(scores)
    td.to_csv(out / "term_dominance.csv", index=False)
    ov.to_csv(out / "alarm_overlap.csv", index=False)

    # summary: evaluate.py rows (with event metrics at the calibrated threshold) + combined detectors
    base = pd.concat([pd.read_csv(out / "cv_rows.csv"), pd.read_csv(out / "baselines.csv")], ignore_index=True)
    comb = rec_rows[rec_rows.variant.str.startswith("combined_")].assign(
        n_events=0, detected=0, false_alarms=0, hours=0.0)  # no calibrated threshold: event columns stay empty
    summarise(pd.concat([base, comb], ignore_index=True)).to_csv(out / "summary.csv", index=False)
    print(f"wrote analysis to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "chbmit")
