"""Export model outputs and explanations as compact JSON for the web visualizer.

    python scripts/export_explanations.py experiment=full     # synthetic, fold-0 model
    python scripts/export_explanations.py experiment=chbmit   # CHB-MIT seizure recordings

For each exported recording segment the JSON holds: channel names and positions, downsampled
traces, the anomaly score and its threshold, the learned directed adjacency per window (quantised),
true labels, and for every detected event its propagation paths, top changed edges, edge-masking
result, Integrated Gradients and temporal attention. Results tables are exported alongside.
Everything is a research output, not a clinical finding.
"""

from __future__ import annotations

import json

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.signal import resample_poly

from neurotrace import DISCLAIMER
from neurotrace.data.splits import by_subject
from neurotrace.eval.scoring import ScoreCalibration, run_recording
from neurotrace.experiment import load_checkpoint, load_dataset, to_jsonable
from neurotrace.explain.attribution import edge_masking, integrated_gradients, temporal_attention
from neurotrace.explain.propagation import detect_events, explain_event
from neurotrace.graph.montage import topomap_positions, unit_sphere_positions
from neurotrace.utils import RESULTS_DIR, ROOT, get_device

TRACE_HZ = 64


def continuous(windows: np.ndarray, sfreq: float, stride_s: float) -> np.ndarray:
    """Rebuild the continuous (C, T) signal from overlapping windows (first ``stride`` of each)."""
    step = int(round(stride_s * sfreq))
    parts = [w[:, :step] for w in windows[:-1]] + [windows[-1]]
    return np.concatenate(parts, axis=1).astype(np.float32)


def q(a: np.ndarray, scale: float, lo: int = -127, hi: int = 127) -> list:
    return np.clip(np.round(np.nan_to_num(a) * scale), lo, hi).astype(int).tolist()


def export_segment(model, rec, cal, thr, lo, hi, device, truth=None) -> dict:
    out = run_recording(model, rec, device)
    score = cal.score(out)
    seg = slice(lo, hi)
    times = rec.times[seg] - rec.times[lo]
    sig = continuous(np.asarray(rec.windows[seg], dtype=np.float32), rec.sfreq, rec.stride_s)
    sig = resample_poly(sig, TRACE_HZ, int(rec.sfreq), axis=1)
    adj = out.adj[seg].astype(np.float32) if out.adj is not None else None
    amax = float(np.percentile(adj, 99.9)) if adj is not None else 1.0
    events = []
    for a, b in detect_events(score[seg], thr):
        ga, gb = lo + a, lo + b
        ex = explain_event(out.adj.astype(np.float32) if out.adj is not None else None, rec.bands, score, ga, gb,
                           rec.times, rec.ch_names, rec.stride_s)
        t_peak = int(ga + np.nanargmax(score[ga:gb]))
        t_ex = max(t_peak, model.cfg.seq_len - 1)
        ig = integrated_gradients(model, rec, t_ex, device)
        att = temporal_attention(model, rec, t_ex, device)
        masks = edge_masking(model, rec, t_ex, [(s, d) for s, d, _ in ex.top_edges[:8]], device)
        t0 = rec.times[lo]
        events.append({
            "start": a, "end": b, "onset_s": float(rec.times[ga] - t0), "end_s": float(rec.times[min(gb, len(rec) - 1)] - t0),
            "peak_score": ex.peak_score, "explained_window": int(t_ex - lo),
            "graph_path": [{"ch": s.channel, "t": s.time_s - t0, "via": s.via, "w": round(s.strength, 4)} for s in ex.graph_path],
            "signal_path": [{"ch": s.channel, "t": s.time_s - t0, "z": round(s.strength, 2)} for s in ex.signal_path],
            "top_edges": [{"src": s, "dst": d, "delta": round(v, 4)} for s, d, v in ex.top_edges],
            "edge_masking": [{"src": s, "dst": d, "effect": round(v, 5)} for s, d, v in masks],
            "ig_bands": np.round(ig["bands"].sum(axis=0), 5).tolist(),  # (C, F) summed over windows
            "ig_channel_time": np.round(ig["bands"].sum(axis=2) + ig["raw"], 5).tolist(),  # (S, C)
            "ig_delta": ig["delta"],
            "attention": np.round(att[-1][-1], 4).tolist() if att is not None else None,  # last layer, last query
        })
    label_runs = []
    lab = rec.labels[seg]
    d = np.diff(np.concatenate([[0], lab.astype(int), [0]]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
        label_runs.append({"start_s": float(times[s]), "end_s": float(times[e - 1] + rec.stride_s)})
    return {
        "subject": rec.subject, "record": rec.record, "offset_s": float(rec.times[lo]),
        "ch_names": rec.ch_names,
        "pos2d": np.round(topomap_positions(rec.ch_names), 4).tolist(),
        "pos3d": np.round(unit_sphere_positions(rec.ch_names), 4).tolist(),
        "stride_s": rec.stride_s, "window_s": rec.window_s, "trace_hz": TRACE_HZ,
        "times": np.round(times, 2).tolist(),
        "score": np.round(np.nan_to_num(score[seg]), 3).tolist(), "threshold": round(thr, 3),
        "components": {k: np.round(cal.z(out, k)[seg], 3).tolist() for k in ("pred_err", "graph_change", "var_err")},
        "adj_scale": amax,
        "adj": [q(a / amax, 99, 0, 99) for a in adj] if adj is not None else None,
        "traces": [q(ch, 25) for ch in sig],  # z-scored signal x 25, clipped to +-127 (about +-5 sd)
        "labels": label_runs, "events": events, "truth": truth,
    }


def results_tables(dataset: str) -> dict:
    p = RESULTS_DIR / dataset / "summary.csv"
    return pd.read_csv(p).replace({np.nan: None}).to_dict(orient="records") if p.exists() else []


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.device)
    recs = load_dataset(cfg)
    name = cfg.data.name
    ck = RESULTS_DIR / name / "models" / "fold0.pt"
    if not ck.exists():
        ck = ROOT / cfg.out_dir / "model.pt"
    model, mcfg = load_checkpoint(ck, recs)
    import torch

    split = torch.load(ck, map_location="cpu", weights_only=False)["split"]
    train_s, val_s, test_s = split
    print(f"model {ck.name}: test subjects {test_s}")
    cal = ScoreCalibration.fit([run_recording(model, r, device) for r in by_subject(recs, val_s)],
                               lambda_graph=mcfg.train.lambda_graph)
    normal = np.concatenate([cal.score(run_recording(model, r, device))[~r.labels] for r in by_subject(recs, val_s)])
    thr = float(np.percentile(normal, cfg.eval.threshold_pct))

    segments = []
    for r in by_subject(recs, test_s):
        if name == "synthetic":
            ev = r.extra["events"]
            truth = {
                "base_edges": np.argwhere(r.extra["base_adj"]).tolist(),
                "events": [{"onset_s": e.onset_s, "offset_s": e.offset_s, "seed": e.seed, "order": e.order,
                            "arrival_s": e.arrival_s, "edges": e.edges} for e in ev],
            }
            segments.append(export_segment(model, r, cal, thr, 0, len(r), device, truth))
        elif r.labels.any():
            on = int(np.argmax(r.labels))
            lo, hi = max(0, on - 240), min(len(r), on + 180)
            segments.append(export_segment(model, r, cal, thr, lo, hi, device))
        print(f"  exported {r.record}")
    out = ROOT / "web" / "public" / "data"
    (out / "seg").mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"segments": [], "summary": {}}
    index["disclaimer"] = DISCLAIMER
    index["edge_semantics"] = "directed statistical dependency (src -> dst), not causation"
    index["bands"] = list(cfg.data.bands.keys())
    index["segments"] = [s for s in index["segments"] if s["dataset"] != name]
    for seg in segments:
        sid = f"{name}-{seg['record']}"
        seg.update({"id": sid, "dataset": name, "model": ck.stem})
        (out / "seg" / f"{sid}.json").write_text(json.dumps(to_jsonable(seg), separators=(",", ":")))
        index["segments"].append({
            "id": sid, "dataset": name, "subject": seg["subject"], "record": seg["record"],
            "duration_s": seg["times"][-1] + seg["window_s"] if seg["times"] else 0,
            "n_detected": len(seg["events"]), "n_labelled": len(seg["labels"]),
        })
    index["summary"][name] = results_tables(name)
    index_path.write_text(json.dumps(to_jsonable(index), indent=1))
    print(f"wrote {len(segments)} segments to {out}")

if __name__ == "__main__":
    main()
