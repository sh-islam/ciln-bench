"""Shared training utilities for journal-edition voter retraining.

Conventions enforced here (do not override per-voter):
- Val split = 10% of the official training set, drawn with seed=0.
- Best val accuracy decides the saved checkpoint; test acc reported is the
  test accuracy at that best-val checkpoint (Option A, val-best test acc).
- Per-epoch metrics logged to a CSV. One row per epoch.
- Final summary written to a JSON when training completes.

Per-voter scripts import build_loaders() and train_voter() and supply
their config via the VoterConfig dataclass.
"""
from __future__ import annotations
import csv
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

JOURNAL_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = JOURNAL_ROOT / "data_cache"
CKPT_DIR = JOURNAL_ROOT / "checkpoints"
LOG_DIR = JOURNAL_ROOT / "logs"
REPORT_DIR = JOURNAL_ROOT / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VAL_SPLIT_SEED = 0    # fixed so the same val set is used across every voter
TRAIN_SEED = 42       # fixed training seed


# -------------------- determinism --------------------

def set_seed(seed: int = TRAIN_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id: int):
    np.random.seed(VAL_SPLIT_SEED + worker_id)


# -------------------- data loading --------------------

SPLITS_ROOT = JOURNAL_ROOT / "output_seed0" / "splits"


def _load_gu_split(dataset: str, split_name: str) -> list[int]:
    """Load index array from the Gu-style 5-way split layout.

    split_name ∈ {"cleanlabeltrain", "noisylabeltrain",
                  "cleanlabelvalid", "noisylabelvalid", "test"}.
    Returns a Python list of int indices into the ORIGINAL dataset.
    """
    f = SPLITS_ROOT / dataset / f"{split_name}_indices.npy"
    if not f.exists():
        raise FileNotFoundError(
            f"Gu-style split index file missing: {f}. "
            f"Run `python splits/build_splits.py` first to generate it."
        )
    return np.load(f).tolist()


def build_loaders(
    dataset: str,
    train_transform,
    eval_transform,
    batch_size: int = 128,
    val_frac: float = 0.10,  # kept for backward-compat; unused (Gu-split is fixed)
    num_workers: int = 4,
):
    """Build train/val/test loaders for voter training in the Gu-style pipeline.

    Voters see:
      - train: CleanLabelTrain (with train_transform / augmentation)
      - val:   CleanLabelValid (with eval_transform)
      - test:  the official Test split (clean, with eval_transform)
    NoisyLabelTrain / NoisyLabelValid are NOT loaded here - they are corrupted
    and consumed downstream by the eval scripts.
    """
    dataset = dataset.lower()
    if dataset == "cifar10":
        full_train = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
        full_train_eval = datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=eval_transform)
        test = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=eval_transform)
    elif dataset == "mnist":
        full_train = datasets.MNIST(root=str(DATA_DIR), train=True, download=True, transform=train_transform)
        full_train_eval = datasets.MNIST(root=str(DATA_DIR), train=True, download=False, transform=eval_transform)
        test = datasets.MNIST(root=str(DATA_DIR), train=False, download=True, transform=eval_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_idx = _load_gu_split(dataset, "cleanlabeltrain")
    val_idx = _load_gu_split(dataset, "cleanlabelvalid")
    train_subset = Subset(full_train, train_idx)
    val_subset = Subset(full_train_eval, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    print(f"[loaders] {dataset}: train={len(train_subset)}  val={len(val_subset)}  test={len(test)}", flush=True)
    return train_loader, val_loader, test_loader


# -------------------- max-norm (for Srivastava MLP) --------------------

def apply_max_norm(model: nn.Module, max_norm: float):
    """Clip the L2 norm of each Linear layer's incoming weight vector (per output unit) to max_norm."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                norms = m.weight.data.norm(p=2, dim=1, keepdim=True)
                desired = norms.clamp(max=max_norm)
                m.weight.data.mul_(desired / (norms + 1e-12))


# -------------------- training loop --------------------

@dataclass
class VoterConfig:
    voter_name: str            # e.g. "lenet5"
    dataset: str               # "mnist" | "cifar10"
    epochs: int
    batch_size: int
    optimizer: str             # "sgd" | "sgd_nesterov" | "adam" | "adamw"
    lr: float
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    scheduler: str = "none"    # "step" | "multistep" | "cosine" | "linear_decay_after" | "none"
    # step / multistep:
    milestones: List[int] = field(default_factory=list)
    gamma: float = 0.1
    # cosine:
    warmup_epochs: int = 0
    warmup_lr: float = 1e-6
    # linear_decay_after (Co-Teaching):
    decay_start_epoch: int = 0
    # regularization:
    label_smoothing: float = 0.0
    max_norm: Optional[float] = None       # e.g. 2.0 for Srivastava MLP
    # data:
    val_frac: float = 0.10
    num_workers: int = 4
    # ckpt:
    save_best_only: bool = True
    notes: str = ""
    # amp:
    use_amp: bool = True   # mixed-precision (FP16 fwd + FP32 master weights via torch.amp)


def build_optimizer(model: nn.Module, cfg: VoterConfig) -> optim.Optimizer:
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_nesterov":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                         weight_decay=cfg.weight_decay, nesterov=True)
    if cfg.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                          weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999),
                           weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def build_scheduler(opt: optim.Optimizer, cfg: VoterConfig):
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(opt, step_size=cfg.milestones[0], gamma=cfg.gamma)
    if cfg.scheduler == "multistep":
        return optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.milestones, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        # Linear warmup, then cosine
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        if cfg.warmup_epochs > 0:
            warmup = LinearLR(opt, start_factor=cfg.warmup_lr / cfg.lr,
                              end_factor=1.0, total_iters=cfg.warmup_epochs)
            main = CosineAnnealingLR(opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=0.0)
            return SequentialLR(opt, [warmup, main], milestones=[cfg.warmup_epochs])
        return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=0.0)
    if cfg.scheduler == "linear_decay_after":
        # Linear decay from full lr at epoch=decay_start to 0 at epoch=epochs.
        def lr_fn(epoch):
            if epoch < cfg.decay_start_epoch:
                return 1.0
            tot = max(1, cfg.epochs - cfg.decay_start_epoch)
            elapsed = epoch - cfg.decay_start_epoch
            return max(0.0, 1.0 - elapsed / tot)
        return optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_fn)
    return None


def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss_sum += crit(logits, y).item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"loss": loss_sum / total, "acc": correct / total}


def train_voter(
    model: nn.Module,
    cfg: VoterConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    train_step_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Generic training loop with per-epoch val eval, best-val checkpoint, test-at-end.

    train_step_fn(model, x, y, criterion, optimizer, cfg) -> loss
        if None, uses a vanilla forward/backward step.
    """
    set_seed(TRAIN_SEED)
    model = model.to(device)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    use_amp = bool(getattr(cfg, "use_amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[{cfg.voter_name}/{cfg.dataset}] AMP enabled: {use_amp}", flush=True)

    if train_step_fn is None:
        def train_step_fn(model, x, y, crit, optim_, cfg_):
            optim_.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optim_)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                optim_.step()
            return loss.item()

    log_path = LOG_DIR / f"{cfg.voter_name}_{cfg.dataset}.csv"
    ckpt_path = CKPT_DIR / f"{cfg.voter_name}_{cfg.dataset}_best.pt"
    summary_path = REPORT_DIR / f"{cfg.voter_name}_{cfg.dataset}.json"

    # CSV header
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "wall_sec"])

    best_val_acc = -1.0
    best_epoch = -1
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        epoch_t0 = time.time()
        model.train()
        n, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = train_step_fn(model, x, y, criterion, opt, cfg)
            if cfg.max_norm is not None:
                apply_max_norm(model, cfg.max_norm)
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                n += y.size(0)
                loss_sum += loss * y.size(0)
        train_loss = loss_sum / max(1, n)
        train_acc = correct / max(1, n)

        val = eval_model(model, val_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        wall = time.time() - epoch_t0
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{lr_now:.6g}", f"{train_loss:.4f}", f"{train_acc:.4f}",
                        f"{val['loss']:.4f}", f"{val['acc']:.4f}", f"{wall:.1f}"])

        print(f"[{cfg.voter_name}/{cfg.dataset}] ep {epoch}/{cfg.epochs}  "
              f"lr={lr_now:.4g}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}  ({wall:.1f}s)", flush=True)

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_acc": val["acc"], "cfg": asdict(cfg)},
                       ckpt_path)

        if sched is not None:
            sched.step()

    # Load best, evaluate on test
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test = eval_model(model, test_loader, device)
    train_eval = eval_model(model, train_loader, device)
    total_wall = time.time() - t0

    summary = {
        "voter_name": cfg.voter_name,
        "dataset": cfg.dataset,
        "best_val_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc_at_best_val": test["acc"],
        "test_loss_at_best_val": test["loss"],
        "train_acc_at_best_val": train_eval["acc"],
        "total_epochs": cfg.epochs,
        "total_wall_sec": total_wall,
        "ckpt_path": str(ckpt_path),
        "log_path": str(log_path),
        "config": asdict(cfg),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== DONE [{cfg.voter_name}/{cfg.dataset}] ===", flush=True)
    print(f"  best val acc:  {best_val_acc:.4f} at epoch {best_epoch}", flush=True)
    print(f"  test acc:      {test['acc']:.4f} (loss {test['loss']:.4f})", flush=True)
    print(f"  train acc:     {train_eval['acc']:.4f}", flush=True)
    print(f"  total time:    {total_wall/60:.1f} min", flush=True)
    print(f"  ckpt:          {ckpt_path}", flush=True)
    return summary
