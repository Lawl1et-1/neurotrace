"""Print results as markdown tables (mean ± 95 % CI half-width, n test subjects).

    python scripts/results_table.py synthetic
    python scripts/results_table.py chbmit            # summary
    python scripts/results_table.py chbmit analysis   # per-patient, Wilcoxon, matched FA, score terms
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from neurotrace.utils import RESULTS_DIR

ORDER = ["full", "diffusion_gnn", "static_graph", "no_graph", "no_transformer", "no_var_evidence",
         "masked_objective", "supervised_head", "channel_cnn_lstm", "baseline_bandpower_dev",
         "baseline_pearson", "baseline_coherence", "baseline_granger", "baseline_pdc", "baseline_dtf",
         "combined_bp_graph", "combined_bp_granger"]
NAMES = {
    "full": "NEUROTRACE (learned graph + GAT + Transformer)", "diffusion_gnn": "– diffusion GNN instead of GAT",
    "static_graph": "– distance-prior graph only", "no_graph": "– no graph", "no_transformer": "– no Transformer",
    "no_var_evidence": "– edge module without VAR evidence", "masked_objective": "– masked-window objective",
    "supervised_head": "– + supervised head (uses labels)", "channel_cnn_lstm": "Per-channel CNN-LSTM",
    "baseline_bandpower_dev": "Band-power deviation", "baseline_pearson": "Pearson deviation",
    "baseline_coherence": "Coherence deviation", "baseline_granger": "Granger (per window)",
    "baseline_pdc": "PDC (per window)", "baseline_dtf": "DTF (per window)",
    "combined_bp_graph": "Band power + learned-graph change", "combined_bp_granger": "Band power + Granger change",
}
SHORT = {"full": "Full", "no_graph": "No graph", "static_graph": "Dist. graph", "supervised_head": "Superv.",
         "baseline_bandpower_dev": "Band power", "baseline_granger": "Granger", "combined_bp_graph": "BP + graph",
         "combined_bp_granger": "BP + Granger"}


def cell(r: pd.Series, m: str, d: int = 2) -> str:
    v, c = r.get(f"{m}_mean", np.nan), r.get(f"{m}_ci95", np.nan)
    if pd.isna(v):
        return "—"
    return f"{v:.{d}f} ± {c:.{d}f}" if not pd.isna(c) else f"{v:.{d}f}"


def main(dataset: str) -> None:
    d = pd.read_csv(RESULTS_DIR / dataset / "summary.csv")
    d["o"] = d["variant"].map({v: i for i, v in enumerate(ORDER)})
    d = d.sort_values("o")
    cols = [("auroc", "Window AUROC", 3), ("auprc", "AUPRC", 3), ("sensitivity", "Sensitivity", 2),
            ("fa_per_hour", "FA / h", 1), ("median_latency_s", "Latency (s)", 1)]
    if dataset == "synthetic":
        cols += [("edge_auroc", "Edge AUROC", 3), ("event_edge_auroc", "Event-edge AUROC", 3), ("path_seed_hit", "Path seed hit", 2)]
    print("| Model | n | " + " | ".join(c[1] for c in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for _, r in d.iterrows():
        print(f"| {NAMES.get(r.variant, r.variant)} | {int(r.n_subjects)} | " + " | ".join(cell(r, m, k) for m, _, k in cols) + " |")


def name(v: str) -> str:
    return NAMES.get(v, v)


def analysis(dataset: str) -> None:
    out = RESULTS_DIR / dataset
    pp = pd.read_csv(out / "per_patient.csv")
    cols = [c for c in SHORT if c in set(pp.variant)]
    wide = pp.pivot(index="subject", columns="variant", values="auroc")[cols]
    print("Per-patient window AUROC\n")
    print("| Patient | " + " | ".join(SHORT[c] for c in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for s, r in wide.iterrows():
        print(f"| {s} | " + " | ".join("—" if pd.isna(r[c]) else f"{r[c]:.3f}" for c in cols) + " |")

    print("\nPaired Wilcoxon signed-rank tests over patients (two-sided)\n")
    print("| Metric | A | B | n | median A − B | A better in | p |")
    print("|---|---|---|---|---|---|---|")
    for _, r in pd.read_csv(out / "wilcoxon.csv").iterrows():
        print(f"| {r.metric.upper()} | {name(r.a)} | {name(r.b)} | {r.n_patients} | {r.median_diff:+.3f} | "
              f"{r.a_better}/{r.n_patients} | {r.p_value:.3g} |")

    mf = pd.read_csv(out / "matched_fa.csv")
    mf["o"] = mf["variant"].map({v: i for i, v in enumerate(ORDER)})
    print("\nSensitivity (latency, s) at matched false-alarm rates; threshold set per test patient (oracle)\n")
    ts = sorted(mf.target_fa_h.unique())
    print("| Detector | " + " | ".join(f"≤ {t:g} FA/h" for t in ts) + " |")
    print("|---|" + "---|" * len(ts))
    for v, d in mf.sort_values("o").groupby("variant", sort=False):
        cells = []
        for t in ts:
            r = d[d.target_fa_h == t].iloc[0]
            cells.append(f"{r.sensitivity_mean:.2f} ± {r.sensitivity_ci95:.2f} ({r.median_latency_s_mean:.0f} s)")
        print(f"| {name(v)} | " + " | ".join(cells) + " |")

    td = pd.read_csv(out / "term_dominance.csv")
    td["o"] = td["variant"].map({v: i for i, v in enumerate(ORDER)})
    print("\nScore terms: window AUROC of each term alone, and its share of the score at alarm windows\n")
    print("| Variant | AUROC pred. error | AUROC graph change | AUROC forecast error | share pred. | share graph | share forecast |")
    print("|---|---|---|---|---|---|---|")
    for _, r in td.sort_values("o").iterrows():
        a = [f"{r[f'auroc_{k}_mean']:.3f}" if pd.notna(r[f"auroc_{k}_mean"]) else "—" for k in ("pred_err", "graph_change", "var_err")]
        sh = [f"{r[f'alarm_share_{k}']:.2f}" for k in ("pred_err", "graph_change", "var_err")]
        print(f"| {name(r.variant)} | " + " | ".join(a + sh) + " |")


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    analysis(ds) if len(sys.argv) > 2 and sys.argv[2] == "analysis" else main(ds)
