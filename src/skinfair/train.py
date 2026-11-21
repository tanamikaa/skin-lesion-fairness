"""Shared training loop for CNN and ViT.

Owners: Anamika Kumari (loop), Khushi Narwariya (ViT schedule)

* AMP (CUDA only), gradient clipping, early stopping on **val balanced accuracy**,
  best-checkpoint saving, per-epoch CSV log.
* Warm-up + cosine LR schedule (per step). `train.warmup_epochs` defaults to 0 (pure cosine),
  the ViT config sets 3.
* ViT: layer-wise LR decay param groups when `model.layer_decay` is set.
* ResNet: backbone frozen for the first `model.freeze_backbone_epochs` epochs.
* Optional mixup (`train.mixup_alpha` > 0), works with every loss.

Config keys: see configs/*.yaml. Each run writes to <output_dir>/<run_name>/ and
<checkpoint_dir>/<run_name>_best.pt, with run_name = <base>_<loss>[_os]_s<seed> unless set.
Extra optional keys: checkpoint_dir (default results/checkpoints), run_name, device, train.grad_clip (default 1.0),
train.mixup_alpha, train.progress (default true).
"""
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from skinfair import NUM_CLASSES
from skinfair.data.datasets import get_dataloaders
from skinfair.evaluation.metrics import fast_balanced_accuracy
from skinfair.losses import build_loss
from skinfair.models import build_model
from skinfair.models.resnet import set_backbone_trainable
from skinfair.models.vit import layer_decay_param_groups
from skinfair.utils import run_dir, run_name

log = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    t = cfg["train"]
    m = cfg["model"]
    if m["name"].startswith("vit") and m.get("layer_decay"):
        groups = layer_decay_param_groups(model, t["lr"], t["weight_decay"], m["layer_decay"])
        return torch.optim.AdamW(groups)
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0, min_lr_ratio=0.01):
    """Linear warm-up for `warmup_epochs`, then cosine decay to `min_lr_ratio` x base LR (per step)."""
    total = max(1, epochs * steps_per_epoch)
    warm = int(warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, total - warm)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def _mixup(images, labels, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[perm], labels, labels[perm], lam


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    losses, ys, ps = [], [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)
        losses.append(criterion(logits.float(), y).item() * len(y))
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return sum(losses) / len(y), fast_balanced_accuracy(y, p), float((y == p).mean())


def train(cfg, loaders=None):
    """Train a model from `cfg`. `loaders` (dict train/val) can be injected, e.g. for tests.

    Returns a summary dict: best_val_bal_acc, best_epoch, checkpoint, log_csv, epochs_run.
    """
    seed_everything(cfg.get("seed", 42))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = cfg["train"]
    name = run_name(cfg)
    out_dir = run_dir(cfg)  # <output_dir>/<run_name>: every run gets its own folder
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}_best.pt"

    loaders = loaders or get_dataloaders(cfg)
    train_loader, val_loader = loaders["train"], loaders["val"]
    counts = np.bincount(train_loader.dataset.labels, minlength=NUM_CLASSES)

    model = build_model(cfg).to(device)
    # Oversampling already balances the classes the model sees; adding class weights on top
    # would double-correct (and hurt the majority classes), so weights are switched off then.
    oversampled = bool(cfg.get("data", {}).get("oversample", False))
    if oversampled and cfg["loss"]["name"] in ("weighted_ce", "focal"):
        log.warning("oversample=true: class weights disabled for loss '%s' (avoid double correction)",
                    cfg["loss"]["name"])
    criterion = build_loss(cfg, class_counts=None if oversampled else counts).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, t["epochs"], len(train_loader), t.get("warmup_epochs", 0))
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0) if cfg["model"]["name"].startswith("resnet") else 0
    grad_clip = t.get("grad_clip", 1.0)
    mixup_alpha = t.get("mixup_alpha", 0.0)

    best, best_epoch, bad, history = -1.0, -1, 0, []
    for epoch in range(t["epochs"]):
        if freeze_epochs:
            set_backbone_trainable(model, epoch >= freeze_epochs)
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        it = tqdm(train_loader, desc=f"epoch {epoch + 1}/{t['epochs']}", leave=False,
                  disable=not t.get("progress", True))
        for batch in it:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                if mixup_alpha > 0:
                    x, ya, yb, lam = _mixup(x, y, mixup_alpha)
                    logits = model(x)
                    loss = lam * criterion(logits.float(), ya) + (1 - lam) * criterion(logits.float(), yb)
                else:
                    loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            run_loss += loss.item() * len(y)
            seen += len(y)

        val_loss, val_bal, val_acc = _validate(model, val_loader, criterion, device)
        row = {"epoch": epoch + 1, "train_loss": run_loss / seen, "val_loss": val_loss,
               "val_bal_acc": val_bal, "val_acc": val_acc, "lr": max(g["lr"] for g in optimizer.param_groups),
               "seconds": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)
        log.info("epoch %d: %s", epoch + 1, {k: round(v, 4) for k, v in row.items()})

        if val_bal > best:
            best, best_epoch, bad = val_bal, epoch + 1, 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "run_name": name, "epoch": epoch + 1,
                        "val_bal_acc": val_bal},
                       ckpt_path)
        else:
            bad += 1
            if bad >= t.get("early_stopping_patience", 5):
                log.info("early stopping at epoch %d (best %d)", epoch + 1, best_epoch)
                break

    summary = {"run": name, "best_val_bal_acc": best, "best_epoch": best_epoch,
               "epochs_run": len(history), "checkpoint": str(ckpt_path), "log_csv": str(out_dir / "log.csv")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
