"""Training engine for the pose models.

This module wraps the numpy-returning dataset in a ``torch.utils.data.Dataset``
adapter, builds the model/optimizer/scheduler, and runs the train/eval loops.

Key rule: ``init_from`` may ONLY point at checkpoints WE have produced inside
``data/processed/stage_*``. No external pretrained weights.
See ``docs/project_decisions.md``.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def _to_tensor(v):
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    return v


class _TorchDatasetAdapter(Dataset):
    """Wrap a numpy-returning dataset into a torch ``Dataset``."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._inner[idx]
        out: Dict = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v.astype(np.float32) if v.dtype != np.uint8 else v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                out[k] = torch.as_tensor(v)
            else:
                out[k] = v  # keep strings / misc untouched
        return out


def _collate(batch):
    out: Dict = {}
    keys = batch[0].keys()
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, torch.Tensor):
            try:
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            except RuntimeError:
                out[k] = [b[k] for b in batch]
        else:
            out[k] = [b[k] for b in batch]
    return out


def build_model(model_config: Dict) -> nn.Module:
    name = model_config.get("name", "").lower()
    if model_config.get("pretrained", False):
        raise ValueError(
            "pretrained=true is forbidden. See docs/project_decisions.md section 1."
        )
    if name == "simple_baseline":
        from src.models.simple_baseline import SimpleBaselinePose
        return SimpleBaselinePose(model_config)
    if name == "hrnet_w32":
        from src.models.hrnet import HRNetPose
        return HRNetPose(model_config)
    raise ValueError(f"Unknown model name: {name!r}")


def build_optimizer(model: nn.Module, optim_cfg: Dict) -> torch.optim.Optimizer:
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg.get("lr", 3e-4))
    wd = float(optim_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name!r}")


def build_scheduler(optim, sched_cfg: Dict, total_epochs: int):
    name = sched_cfg.get("name", "cosine").lower()
    warmup = int(sched_cfg.get("warmup_epochs", 0))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    base_lrs = [g["lr"] for g in optim.param_groups]

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        if name == "cosine":
            progress = (epoch - warmup) / max(total_epochs - warmup, 1)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
            return max(scale + (min_lr / base_lrs[0]) * (1 - scale), min_lr / base_lrs[0])
        if name == "constant":
            return 1.0
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=_lr_lambda)


@dataclass
class TrainCtx:
    device: torch.device
    model: nn.Module
    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    loss_fn: nn.Module
    output_dir: Path
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 2
    best_metric: float = -float("inf")


def _load_state_from_internal_ckpt(model: nn.Module, path: str) -> None:
    """Load WEIGHTS from an internal checkpoint (produced by this project).

    Safety: the path must live under ``data/processed/`` -- never from the web,
    never ImageNet, etc. (See docs/project_decisions.md.)
    """
    normalized = os.path.normpath(path)
    assert "data/processed/" in normalized.replace("\\", "/"), (
        f"init_from must point INSIDE data/processed/ (ours). Got: {path}. "
        f"See docs/project_decisions.md section 1."
    )
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[init_from] missing keys (ok if architecture differs slightly): {len(missing)}")
    if unexpected:
        print(f"[init_from] unexpected keys: {len(unexpected)}")


def make_train_ctx(
    train_cfg: Dict,
    model_cfg: Dict,
    device: Optional[torch.device] = None,
) -> TrainCtx:
    from src.models.losses import JointMSELoss

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)

    if (init_from := train_cfg.get("init_from")):
        _load_state_from_internal_ckpt(model, init_from)

    optim = build_optimizer(model, train_cfg.get("optimizer", {}))
    scheduler = build_scheduler(optim, train_cfg.get("scheduler", {}), int(train_cfg.get("epochs", 1)))

    loss_cfg = train_cfg.get("loss", {})
    loss_fn = JointMSELoss(use_target_weight=bool(loss_cfg.get("use_target_weight", True))).to(device)

    return TrainCtx(
        device=device,
        model=model,
        optim=optim,
        scheduler=scheduler,
        loss_fn=loss_fn,
        output_dir=Path(train_cfg["output_dir"]),
        log_every_n_steps=int(train_cfg.get("log_every_n_steps", 50)),
        save_every_n_epochs=int(train_cfg.get("save_every_n_epochs", 5)),
        eval_every_n_epochs=int(train_cfg.get("eval_every_n_epochs", 2)),
    )


def make_loader(numpy_dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        _TorchDatasetAdapter(numpy_dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=shuffle,
        pin_memory=False,
    )


def train_one_epoch(ctx: TrainCtx, loader: DataLoader, epoch: int) -> Dict[str, float]:
    ctx.model.train()
    running = 0.0
    n_steps = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(ctx.device)
        targets = batch["heatmaps"].to(ctx.device)
        weights = batch["target_weight"].to(ctx.device)
        preds = ctx.model(images)
        loss = ctx.loss_fn(preds, targets, weights)
        ctx.optim.zero_grad(set_to_none=True)
        loss.backward()
        ctx.optim.step()
        running += float(loss.item())
        n_steps += 1
        if step % ctx.log_every_n_steps == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.5f}")
    ctx.scheduler.step()
    return {"train_loss": running / max(n_steps, 1)}


@torch.no_grad()
def evaluate(ctx: TrainCtx, loader: DataLoader) -> Dict[str, float]:
    """Fast evaluation: heatmap MSE + argmax PCK-ish on heatmap coords."""
    from src.models.decode import argmax_heatmaps
    ctx.model.eval()
    losses: list[float] = []
    pcks: list[float] = []
    for batch in loader:
        images = batch["image"].to(ctx.device)
        targets = batch["heatmaps"].to(ctx.device)
        weights = batch["target_weight"].to(ctx.device)
        preds = ctx.model(images)
        loss = ctx.loss_fn(preds, targets, weights)
        losses.append(float(loss.item()))

        # PCK@0.1 of heatmap diagonal: how many joints have argmax <= 10% of heatmap diag from gt.
        coords_pred, _ = argmax_heatmaps(preds)
        coords_gt = batch["kps_hm"].to(ctx.device)
        diag = float((targets.shape[-1] ** 2 + targets.shape[-2] ** 2) ** 0.5)
        d = (coords_pred - coords_gt).norm(dim=-1)
        w = weights.squeeze(-1)
        mask = w > 0
        if mask.any():
            pcks.append(float(((d <= 0.1 * diag) & mask).sum().item() / mask.sum().item()))

    out = {"val_loss": float(np.mean(losses)) if losses else float("nan")}
    out["val_pck01"] = float(np.mean(pcks)) if pcks else float("nan")
    return out


def save_checkpoint(ctx: TrainCtx, path: Path, extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": ctx.model.state_dict()}
    if extra:
        state.update(extra)
    torch.save(state, path)
