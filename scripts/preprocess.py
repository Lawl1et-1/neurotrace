"""Filter, normalise and window downloaded CHB-MIT files (one cached .npz per EDF).

    python scripts/preprocess.py data=chbmit
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from neurotrace.data.datasets.chbmit import preprocess_subject
from neurotrace.data.preprocess import PreprocessConfig
from neurotrace.utils import ROOT


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    d = cfg.data
    pcfg = PreprocessConfig(sfreq=d.sfreq, l_freq=d.l_freq, h_freq=d.h_freq, notch=d.notch)
    bands = {k: tuple(v) for k, v in d.bands.items()}
    for s in d.subjects:
        print(f"[{s}]")
        preprocess_subject(s, ROOT / d.raw_dir, ROOT / d.cache_dir, pcfg, d.window_s, d.stride_s, bands)


if __name__ == "__main__":
    main()
