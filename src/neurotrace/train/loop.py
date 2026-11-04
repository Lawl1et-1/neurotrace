"""Training loop for NEUROTRACE (self-supervised, optionally with a supervised seizure head)."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from neurotrace.models.neurotrace import NeuroTrace


@dataclass
class TrainConfig:
    objective: str = "next"
    mask_ratio: float = 0.25
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    lambda_sparse: float = 1e-3
    lambda_var: float = 1.0
    supervised: bool = False
    patience: int = 6
    max_batches: int | None = None  # cap batches per epoch (quick runs)

    @classmethod
    def from_cfg(cls, cfg) -> TrainConfig:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in dict(cfg).items() if k in keys})


def compute_loss(model: NeuroTrace, batch: dict[str, torch.Tensor], tc: TrainConfig) -> dict[str, torch.Tensor]:
    """All loss terms for one batch. Returns a dict with ``total`` and each component."""
    x, bands = batch["x"], batch["bands"]
    mask = None
    if tc.objective == "masked":
        mask = torch.rand(x.shape[:2], device=x.device) < tc.mask_ratio
        mask[:, 0] |= ~mask.any(dim=1)  # at least one masked window per sequence
    out = model(x, bands, mask)
    target = out["bands_n"]
    if tc.objective == "next":
        pred_loss = ((out["pred"][:, :-1] - target[:, 1:]) ** 2).mean()
    else:
        pred_loss = ((out["pred"] - target) ** 2).mean(dim=(2, 3))[mask].mean()
    losses = {"pred": pred_loss}
    if "var_err" in out:
        losses["var"] = tc.lambda_var * out["var_err"].mean()
        losses["sparse"] = tc.lambda_sparse * out["adj"].mean() * out["adj"].shape[-1]
    if tc.supervised and "logits" in out:
        y = batch["labels"]
        pos = y.mean().clamp(1e-3, 0.5)
        w = torch.where(y > 0.5, (1 - pos) / pos, torch.ones_like(y))
        losses["bce"] = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], y, weight=w)
    losses["total"] = sum(losses.values())
    return losses


def _to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def train_model(
    model: NeuroTrace,
    train_ds,
    val_ds,
    tc: TrainConfig,
    device: str = "cpu",
    num_workers: int = 0,
    log=print,
) -> tuple[NeuroTrace, list[dict]]:
    """Train with AdamW + cosine schedule, early stopping on validation loss; returns best model."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps = tc.epochs * (min(len(train_ds) // tc.batch_size + 1, tc.max_batches or 10**9))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=max(steps, 1), pct_start=0.1)
    gen = torch.Generator().manual_seed(0)
    tl = DataLoader(train_ds, tc.batch_size, shuffle=True, num_workers=num_workers, generator=gen, drop_last=len(train_ds) > tc.batch_size)
    vl = DataLoader(val_ds, tc.batch_size, shuffle=False, num_workers=num_workers)
    best, best_loss, bad, history = copy.deepcopy(model.state_dict()), float("inf"), 0, []
    for epoch in range(tc.epochs):
        t0 = time.time()
        model.train()
        agg: dict[str, float] = {}
        nb = 0
        for i, batch in enumerate(tl):
            if tc.max_batches and i >= tc.max_batches:
                break
            losses = compute_loss(model, _to(batch, device), tc)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + float(v)
            nb += 1
        val = evaluate_loss(model, vl, tc, device)
        rec = {"epoch": epoch, **{f"train_{k}": v / max(nb, 1) for k, v in agg.items()}, **{f"val_{k}": v for k, v in val.items()}, "sec": time.time() - t0}
        history.append(rec)
        log(f"epoch {epoch:3d}  train {rec.get('train_total', 0):.4f}  val {val['total']:.4f}  ({rec['sec']:.0f}s)")
        if val["total"] < best_loss - 1e-4:
            best_loss, bad, best = val["total"], 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tc.patience:
                log("early stopping")
                break
    model.load_state_dict(best)
    return model, history


@torch.no_grad()
def evaluate_loss(model: NeuroTrace, loader, tc: TrainConfig, device: str) -> dict[str, float]:
    """Mean loss terms over a loader (eval mode, fixed mask seed for the masked objective)."""
    model.eval()
    torch.manual_seed(0)
    agg: dict[str, float] = {}
    n = 0
    for batch in loader:
        for k, v in compute_loss(model, _to(batch, device), tc).items():
            agg[k] = agg.get(k, 0.0) + float(v)
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}
