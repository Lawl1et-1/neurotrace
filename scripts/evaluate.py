"""Cross-validated evaluation of NEUROTRACE, its ablations and the baselines.

    python scripts/evaluate.py experiment=full            # synthetic, 4 folds
    python scripts/evaluate.py experiment=chbmit          # CHB-MIT, leave-one-subject-out
    python scripts/evaluate.py experiment=smoke eval.variants=[full,no_graph]

Writes ``results/<dataset>/``: ``cv_rows.csv`` (one row per variant x test recording),
``baselines.csv``, ``summary.csv`` (mean/std over subjects) and figures.
"""

from __future__ import annotations

import hydra
import pandas as pd
from omegaconf import DictConfig

from neurotrace.eval.plots import plot_summary
from neurotrace.eval.runner import classical_baselines, run_cv, summarise
from neurotrace.experiment import load_dataset
from neurotrace.utils import RESULTS_DIR, get_device, seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = RESULTS_DIR / cfg.data.name
    out.mkdir(parents=True, exist_ok=True)
    recs = load_dataset(cfg)
    print(f"{len(recs)} recordings, {len({r.subject for r in recs})} subjects, device {device}")

    base = classical_baselines(cfg, recs, list(cfg.eval.baseline_methods))
    base.to_csv(out / "baselines.csv", index=False)
    cv = run_cv(cfg, recs, device, out, list(cfg.eval.variants))
    summary = summarise(pd.concat([cv, base], ignore_index=True))
    summary.to_csv(out / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3))
    plot_summary(summary, out, cfg.data.name)


if __name__ == "__main__":
    main()
