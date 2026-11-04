"""Result figures (matplotlib, written to ``results/<dataset>/figures``)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ACCENT = "#2a6f97"
MUTED = "#9aa5b1"
BASELINE = "#c9a227"


def _bar(ax, df: pd.DataFrame, metric: str, title: str, chance: float | None = None) -> None:
    d = df.dropna(subset=[f"{metric}_mean"]).sort_values(f"{metric}_mean")
    colors = [ACCENT if v == "full" else BASELINE if v.startswith("baseline_") else MUTED for v in d["variant"]]
    ax.barh(d["variant"], d[f"{metric}_mean"], xerr=d[f"{metric}_std"].fillna(0), color=colors,
            error_kw={"elinewidth": 0.8, "ecolor": "#444"})
    if chance is not None:
        ax.axvline(chance, color="#444", lw=0.8, ls="--")
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def plot_summary(summary: pd.DataFrame, out: Path, dataset: str) -> None:
    """Window AUROC / AUPRC per variant and, on synthetic data, edge recovery."""
    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panels = [("auroc", "Window AUROC (anomaly score vs label)", 0.5), ("auprc", "Window AUPRC", None)]
    if "edge_auroc_mean" in summary and summary["edge_auroc_mean"].notna().any():
        panels += [("edge_auroc", "Edge AUROC (directed couplings)", 0.5),
                   ("event_edge_auroc", "Event-edge AUROC (graph change)", 0.5)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), constrained_layout=True)
    for ax, (m, t, ch) in zip(axes, panels, strict=True):
        if f"{m}_mean" in summary:
            _bar(ax, summary, m, t, ch)
    fig.suptitle(f"NEUROTRACE - {dataset} (mean ± sd over test subjects). Research prototype, not for clinical use.",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(fig_dir / "summary.png", dpi=150)
    plt.close(fig)
