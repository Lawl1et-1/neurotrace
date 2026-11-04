"""Print a results summary as a markdown table (mean ± 95 % CI half-width, n test subjects).

    python scripts/results_table.py synthetic
    python scripts/results_table.py chbmit
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from neurotrace.utils import RESULTS_DIR

ORDER = ["full", "diffusion_gnn", "static_graph", "no_graph", "no_transformer", "no_var_evidence",
         "masked_objective", "supervised_head", "channel_cnn_lstm", "baseline_bandpower_dev",
         "baseline_pearson", "baseline_coherence", "baseline_granger", "baseline_pdc", "baseline_dtf"]
NAMES = {
    "full": "NEUROTRACE (learned graph + GAT + Transformer)", "diffusion_gnn": "– diffusion GNN instead of GAT",
    "static_graph": "– distance-prior graph only", "no_graph": "– no graph", "no_transformer": "– no Transformer",
    "no_var_evidence": "– edge module without VAR evidence", "masked_objective": "– masked-window objective",
    "supervised_head": "– + supervised head (uses labels)", "channel_cnn_lstm": "Per-channel CNN-LSTM",
    "baseline_bandpower_dev": "Band-power deviation", "baseline_pearson": "Pearson deviation",
    "baseline_coherence": "Coherence deviation", "baseline_granger": "Granger (per window)",
    "baseline_pdc": "PDC (per window)", "baseline_dtf": "DTF (per window)",
}


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


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
