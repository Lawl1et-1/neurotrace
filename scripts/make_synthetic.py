"""Generate the synthetic cohort and print its ground truth (sanity check, nothing is cached).

    python scripts/make_synthetic.py experiment=smoke
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from neurotrace.data.datasets.synthetic_ds import load_synthetic
from neurotrace.utils import seed_everything


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    recs = load_synthetic(cfg.data, seed=cfg.seed)
    for r in recs:
        ev = r.extra["events"]
        print(
            f"{r.subject}: {len(r)} windows {r.windows.shape[1:]}, {int(r.labels.sum())} event windows, "
            f"{int(r.extra['base_adj'].sum())} baseline edges, seeds {[r.ch_names[e.seed] for e in ev]}"
        )


if __name__ == "__main__":
    main()
