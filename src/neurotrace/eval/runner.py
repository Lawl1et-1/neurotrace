"""Cross-validated evaluation: NEUROTRACE ablations, neural and classical baselines.

For every leave-subjects-out fold and every variant:

1. train on the training subjects (self-supervised, normal windows only, unless supervised),
2. calibrate the anomaly score on the validation subject(s): robust z-scores of each component and
   an alarm threshold at the ``eval.threshold_pct`` percentile of *normal* validation scores,
3. score the test subjects: window AUROC/AUPRC, event sensitivity, latency, false alarms per hour,
   and on synthetic data the edge recovery and propagation-path agreement.

Classical baselines need no training and are computed once per subject.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from neurotrace.baselines.connectivity import baseline_deviation, connectivity
from neurotrace.baselines.neural import channel_baseline_errors
from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.data.splits import by_subject, leave_subjects_out
from neurotrace.eval.metrics import edge_auroc, event_metrics, window_metrics
from neurotrace.eval.scoring import RecordingOutput, ScoreCalibration, causal_smooth, run_recording
from neurotrace.experiment import fit, save_checkpoint
from neurotrace.explain.propagation import delta_graph, graph_path, path_agreement

# name -> config overrides (dotlist). "full" is the default model.
VARIANTS: dict[str, list[str]] = {
    "full": [],
    "diffusion_gnn": ["model.spatial=diffusion"],
    "static_graph": ["model.graph=static"],
    "no_graph": ["model.graph=none"],
    "no_transformer": ["model.temporal=none"],
    "no_var_evidence": ["model.var_order=0"],
    "masked_objective": ["train.objective=masked"],
    "supervised_head": ["train.supervised=true"],
    "channel_cnn_lstm": ["model.name=channel_lstm"],
}


def variant_cfg(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    c = copy.deepcopy(cfg)
    OmegaConf.set_struct(c, False)
    return OmegaConf.merge(c, OmegaConf.from_dotlist(overrides))


def _score_outputs(model, recs: list[WindowedRecording], device: str) -> list[RecordingOutput]:
    if model.__class__.__name__ == "ChannelCNNLSTM":
        outs = []
        for r in recs:
            e = channel_baseline_errors(model, r, device)
            z = np.zeros(len(r), np.float32)
            outs.append(RecordingOutput(r.subject, r.record, r.times, r.labels, e, z, z, None))
        return outs
    return [run_recording(model, r, device) for r in recs]


def _robust_z(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = ref[np.isfinite(ref)]
    med = np.median(ref)
    iqr = np.subtract(*np.percentile(ref, [75, 25])) or 1.0
    return np.nan_to_num((v - med) / iqr)


def _trailing_z(v: np.ndarray, n: int) -> np.ndarray:
    """Causal robust z-score against the preceding ``n`` values (see ScoreCalibration.z)."""
    past = pd.Series(v.astype(np.float64)).shift(1).rolling(n, min_periods=30)
    iqr = (past.quantile(0.75) - past.quantile(0.25)).to_numpy()
    iqr = np.where(np.isfinite(iqr) & (iqr > 1e-8), iqr, np.nan)
    return np.nan_to_num((v - past.median().to_numpy()) / iqr, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_recordings(name: str, fold: int, scores: list[np.ndarray], outs_or_recs, threshold: float,
                        stride_s: float, extra_fn=None) -> list[dict]:
    rows = []
    for sc, o in zip(scores, outs_or_recs, strict=True):
        row = {"variant": name, "fold": fold, "subject": o.subject, "record": o.record}
        row.update(window_metrics(sc, o.labels))
        row.update(event_metrics(sc, o.labels, threshold, stride_s))
        if extra_fn is not None:
            row.update(extra_fn(o))
        rows.append(row)
    return rows


def synthetic_extras(rec: WindowedRecording, adj: np.ndarray | None) -> dict:
    """Edge AUROC, event-edge recovery and path agreement for one subject.

    Paths are scored with two seed rules (``net`` and ``hybrid``, see ``choose_seed``) so the
    learned graph and classical estimators are compared on the same footing.
    """
    if adj is None or "adj" not in rec.extra:
        return {}
    a = adj.astype(np.float32)
    out = {"edge_auroc": edge_auroc(a, rec.extra["adj"])}
    ev_auc = []
    agg: dict[str, list[float]] = {}
    for ev in rec.extra["events"]:
        lo, hi = int(ev.onset_s / rec.stride_s) + 1, int(ev.offset_s / rec.stride_s) - 1
        t = np.zeros(a.shape[1:], bool)
        for s, d in ev.edges:
            t[s, d] = True
        ev_auc.append(edge_auroc(delta_graph(a, lo, hi, stride_s=rec.stride_s), t))
        for rule in ("net", "hybrid"):
            steps, _ = graph_path(a, lo, hi, rec.times, rec.ch_names, stride_s=rec.stride_s,
                                  max_steps=len(ev.order), seed_rule=rule, bands=rec.bands)
            pa = path_agreement([s.channel for s in steps], ev.order)
            sfx = "" if rule == "hybrid" else "_net"
            for k, v in pa.items():
                agg.setdefault(f"path_{k}{sfx}", []).append(v)
    if ev_auc:
        out["event_edge_auroc"] = float(np.nanmean(ev_auc))
        for k, v in agg.items():
            out[k] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def run_cv(cfg: DictConfig, recs: list[WindowedRecording], device: str, out_dir: Path,
           variants: list[str], log=print) -> pd.DataFrame:
    """Train/evaluate every variant on every fold; returns one row per (variant, fold, recording)."""
    ev = cfg.eval
    rows: list[dict] = []
    stride_s = recs[0].stride_s
    folds = list(leave_subjects_out(recs, n_folds=ev.n_folds, val_subjects=ev.val_subjects, seed=cfg.seed))
    for fi, (train_s, val_s, test_s) in enumerate(folds):
        log(f"=== fold {fi}: test {test_s} val {val_s}")
        for name in variants:
            vc = variant_cfg(cfg, VARIANTS[name])
            t0 = time.time()
            model, hist = fit(vc, by_subject(recs, train_s), by_subject(recs, val_s), device, log=lambda *_: None)
            val_out = _score_outputs(model, by_subject(recs, val_s), device)
            test_recs = by_subject(recs, test_s)
            test_out = _score_outputs(model, test_recs, device)
            use_prob = name == "supervised_head" and test_out[0].prob is not None
            if use_prob:
                val_scores = [causal_smooth(np.nan_to_num(o.prob), 5) for o in val_out]
                scores = [causal_smooth(np.nan_to_num(o.prob), 5) for o in test_out]
            else:
                cal = ScoreCalibration.fit(val_out, lambda_graph=vc.train.lambda_graph, adaptive=int(ev.get("adaptive_norm", 0)))
                val_scores = [cal.score(o) for o in val_out]
                scores = [cal.score(o) for o in test_out]
            normal = np.concatenate([s[~o.labels] for s, o in zip(val_scores, val_out, strict=True)])
            thr = float(np.percentile(normal, ev.threshold_pct))
            by_rec = {r.record: r for r in test_recs}
            rows += evaluate_recordings(
                name, fi, scores, test_out, thr, stride_s,
                lambda o, by_rec=by_rec: synthetic_extras(by_rec[o.record], o.adj),
            )
            last = pd.DataFrame([r for r in rows if r["variant"] == name and r["fold"] == fi])
            log(f"  {name:18s} auroc {last['auroc'].mean():.3f} auprc {last['auprc'].mean():.3f} "
                f"sens {last['sensitivity'].mean():.2f} fa/h {last['fa_per_hour'].mean():.2f}"
                + (f" edge {last['edge_auroc'].mean():.3f}" if "edge_auroc" in last else "")
                + f" ({time.time() - t0:.0f}s, {len(hist)} ep)")
            if name == "full":
                save_checkpoint(model, vc, out_dir / "models" / f"fold{fi}.pt", {"split": [train_s, val_s, test_s]})
            pd.DataFrame(rows).to_csv(out_dir / "cv_rows.csv", index=False)
    return pd.DataFrame(rows)


def classical_baselines(cfg: DictConfig, recs: list[WindowedRecording], methods: list[str],
                        log=print) -> pd.DataFrame:
    """Connectivity-deviation and band-power-deviation anomaly scores, plus edge recovery (synthetic).

    The connectivity score is the distance of the current graph from the preceding minute's graph
    (``baseline_deviation``), the same quantity the model's own graph term uses.

    The threshold is the ``threshold_pct`` percentile of the subject's own normal windows *from a
    different subject* (the next one in the list) so it is not tuned on the test subject.
    """
    ev = cfg.eval
    every = int(ev.get("baseline_every", 1))
    per: dict[str, dict] = {}
    for r in recs:
        res = {}
        idx = np.arange(0, len(r), every)
        for m in methods:
            a = connectivity(r.windows[idx].astype(np.float64), r.sfreq, m)
            sc = np.interp(np.arange(len(r)), idx, baseline_deviation(a, lo=60 // every, hi=max(1, 5 // every)))
            res[m] = (sc, a, idx)
        # band power deviation: how far the window's band powers are from the recording's median
        b = r.bands.reshape(len(r), -1)
        med = np.median(b, axis=0)
        iqr = np.subtract(*np.percentile(b, [75, 25], axis=0)) + 1e-6
        res["bandpower_dev"] = (np.abs((b - med) / iqr).mean(axis=1), None, idx)
        per[r.record] = res
        log(f"  baselines {r.record}")
    rows = []
    rec_list = list(recs)
    for i, r in enumerate(rec_list):
        ref = rec_list[(i + 1) % len(rec_list)]
        for m, (sc, a, _idx) in per[r.record].items():
            ref_sc = per[ref.record][m][0]
            if ev.get("adaptive_norm", 0):
                z = causal_smooth(_trailing_z(sc, int(ev.adaptive_norm)), 5)
                zref = causal_smooth(_trailing_z(ref_sc, int(ev.adaptive_norm)), 5)
            else:
                z = causal_smooth(_robust_z(sc, ref_sc[~ref.labels]), 5)
                zref = causal_smooth(_robust_z(ref_sc, ref_sc[~ref.labels]), 5)
            thr = float(np.percentile(zref[~ref.labels], ev.threshold_pct))
            row = {"variant": f"baseline_{m}", "fold": -1, "subject": r.subject, "record": r.record}
            row.update(window_metrics(z, r.labels))
            row.update(event_metrics(z, r.labels, thr, r.stride_s))
            if a is not None and "adj" in r.extra:
                full = np.repeat(a, every, axis=0)[: len(r)] if every > 1 else a
                row.update(synthetic_extras(r, full))
            rows.append(row)
    return pd.DataFrame(rows)


METRICS = ["auroc", "auprc", "sensitivity", "median_latency_s", "fa_per_hour", "edge_auroc",
           "event_edge_auroc", "path_seed_hit", "path_recall", "path_kendall",
           "path_seed_hit_net", "path_recall_net", "path_kendall_net"]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per variant. Each metric gets ``_mean``, ``_std``, ``_n`` (test subjects with a
    defined value) and ``_ci95`` (half-width of a t-interval over subjects). Subjects, not windows
    or recordings, are the unit: event counts and hours are pooled within a subject first.
    """
    from scipy import stats

    g = df.groupby(["variant", "subject"])
    subj = g.mean(numeric_only=True).reset_index()
    pooled = g[["detected", "n_events", "false_alarms", "hours"]].sum().reset_index()
    subj["sensitivity"] = pooled["detected"] / pooled["n_events"].replace(0, np.nan)
    subj["fa_per_hour"] = pooled["false_alarms"] / pooled["hours"].replace(0, np.nan)
    rows = []
    for variant, d in subj.groupby("variant"):
        row: dict = {"variant": variant, "n_subjects": len(d)}
        for m in METRICS:
            if m not in d:
                continue
            v = d[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            n = len(v)
            row[f"{m}_mean"] = float(v.mean()) if n else np.nan
            row[f"{m}_std"] = float(v.std(ddof=1)) if n > 1 else np.nan
            row[f"{m}_n"] = n
            row[f"{m}_ci95"] = float(stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
