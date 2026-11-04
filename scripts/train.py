"""Train one model on one subject-level split and save it.

    python scripts/train.py experiment=smoke                       # tiny CPU run
    python scripts/train.py experiment=full model.graph=static     # an ablation
    python scripts/train.py data=chbmit experiment=chbmit

The first fold of leave-subjects-out is used (test subjects are never seen during training).
Full cross-validation and the ablation table are produced by ``scripts/evaluate.py``.
"""

from __future__ import annotations

import json

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from neurotrace.data.splits import by_subject, leave_subjects_out
from neurotrace.eval.metrics import edge_auroc, window_metrics
from neurotrace.eval.scoring import ScoreCalibration, run_recording
from neurotrace.experiment import fit, load_dataset, save_checkpoint
from neurotrace.utils import ROOT, get_device


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.device)
    print(OmegaConf.to_yaml(cfg))
    recs = load_dataset(cfg)
    n_folds = cfg.get("eval", {}).get("n_folds", 4) if "eval" in cfg else 4
    train_s, val_s, test_s = next(leave_subjects_out(recs, n_folds=n_folds, seed=cfg.seed))
    print(f"train {train_s}\nval {val_s}\ntest {test_s}")
    model, history = fit(cfg, by_subject(recs, train_s), by_subject(recs, val_s), device)

    out_dir = ROOT / cfg.out_dir
    save_checkpoint(model, cfg, out_dir / "model.pt", {"split": [train_s, val_s, test_s]})
    val_out = [run_recording(model, r, device) for r in by_subject(recs, val_s)]
    cal = ScoreCalibration.fit(val_out, lambda_graph=cfg.train.lambda_graph)
    summary = {"history": history}
    for r in by_subject(recs, test_s):
        o = run_recording(model, r, device)
        m = window_metrics(cal.score(o), o.labels)
        if o.adj is not None and "adj" in r.extra:
            m["edge_auroc"] = edge_auroc(o.adj.astype(np.float32), r.extra["adj"])
        print(r.record, {k: round(v, 3) for k, v in m.items()})
        summary[r.record] = m
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
