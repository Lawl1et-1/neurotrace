"""Glue between configs, datasets, models and training (used by the scripts in ``scripts/``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from neurotrace.baselines.neural import ChannelBaselineConfig, ChannelCNNLSTM
from neurotrace.data.datasets.base import WindowedRecording
from neurotrace.models.neurotrace import ModelConfig, NeuroTrace
from neurotrace.train.dataset import SequenceDataset, band_stats
from neurotrace.train.loop import TrainConfig, train_model
from neurotrace.utils import ROOT, seed_everything


def load_dataset(cfg: DictConfig) -> list[WindowedRecording]:
    """Load the dataset named in ``cfg.data.name``."""
    name = cfg.data.name
    if name == "synthetic":
        from neurotrace.data.datasets.synthetic_ds import load_synthetic

        return load_synthetic(cfg.data, seed=cfg.seed)
    if name == "chbmit":
        from neurotrace.data.datasets.chbmit import load_chbmit

        recs = load_chbmit(ROOT / cfg.data.cache_dir, list(cfg.data.subjects))
        if not recs:
            raise FileNotFoundError("no preprocessed CHB-MIT files; run `make data` first")
        return recs
    raise ValueError(f"unknown dataset {name!r}")


def build_model(cfg: DictConfig, recs: list[WindowedRecording]) -> torch.nn.Module:
    """Instantiate NEUROTRACE (or the channel-wise baseline when ``model.name == channel_lstm``)."""
    r0 = recs[0]
    c, length, nb = r0.windows.shape[1], r0.windows.shape[2], r0.bands.shape[2]
    if cfg.model.get("name") == "channel_lstm":
        return ChannelCNNLSTM(ChannelBaselineConfig(c, length, nb, cfg.model.hidden, cfg.model.seq_len))
    mc = ModelConfig.from_cfg(cfg.model, cfg.train, c, length, nb)
    return NeuroTrace(mc, r0.ch_names)


def fit(
    cfg: DictConfig,
    train_recs: list[WindowedRecording],
    val_recs: list[WindowedRecording],
    device: str,
    log=print,
) -> tuple[torch.nn.Module, list[dict]]:
    """Seed, build, standardise and train a model on the given subjects."""
    seed_everything(cfg.seed)
    model = build_model(cfg, train_recs)
    mean, std = band_stats(train_recs)
    model.set_band_stats(mean, std)
    seq_len = cfg.model.seq_len
    normal_only = cfg.train.get("train_on", "normal") == "normal" and not cfg.train.supervised
    stride = int(cfg.train.get("seq_stride", 1))
    train_ds = SequenceDataset(train_recs, seq_len, normal_only=normal_only, stride=stride)
    val_ds = SequenceDataset(val_recs, seq_len, normal_only=normal_only, stride=max(stride, seq_len // 2))
    log(f"train sequences: {len(train_ds)}, val sequences: {len(val_ds)}")
    tc = TrainConfig.from_cfg(cfg.train)
    return train_model(model, train_ds, val_ds, tc, device, cfg.train.get("num_workers", 0), log=log)


def save_checkpoint(model: torch.nn.Module, cfg: DictConfig, path: str | Path, extra: dict | None = None) -> None:
    """Weights + resolved config, so a run can be re-loaded without the original command line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True), **(extra or {})}, path)


def load_checkpoint(path: str | Path, recs: list[WindowedRecording]) -> tuple[torch.nn.Module, DictConfig]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["cfg"])
    model = build_model(cfg, recs)
    model.load_state_dict(ck["state_dict"])
    return model.eval(), cfg


def to_jsonable(x):
    """Convert numpy scalars/arrays inside nested structures to plain Python for JSON output."""
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x
