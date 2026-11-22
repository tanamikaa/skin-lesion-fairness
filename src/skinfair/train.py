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
    """Seed Python, NumPy and PyTorch (CPU + all GPUs) RNGs so runs are repeatable.

    Input: integer seed. Returns nothing.
    """
    random.seed(seed)
    np.random.seed(seed)  # also drives the mixup lambda (np.random.beta)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, cfg):
    """Create an AdamW optimizer for `model`.

    Reads train.lr, train.weight_decay, model.name and model.layer_decay. For ViT with
    layer_decay set, uses per-layer param groups (different LR per block); otherwise one
    group with the same LR and weight decay for all parameters. Returns the optimizer.
    """
    t = cfg["train"]
    m = cfg["model"]
    if m["name"].startswith("vit") and m.get("layer_decay"):
        groups = layer_decay_param_groups(model, t["lr"], t["weight_decay"], m["layer_decay"])
        return torch.optim.AdamW(groups)  # each group already carries its own lr / weight_decay
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0, min_lr_ratio=0.01):
    """Linear warm-up for `warmup_epochs`, then cosine decay to `min_lr_ratio` x base LR (per step).

    Inputs: optimizer, number of epochs, batches per epoch, warm-up epochs, final LR ratio.
    Returns a LambdaLR scheduler; call scheduler.step() once after every optimizer step.
    """
    total = max(1, epochs * steps_per_epoch)  # total optimizer steps (at least 1)
    warm = int(warmup_epochs * steps_per_epoch)  # number of warm-up steps

    def lr_lambda(step):
        # Returns the LR multiplier (factor times each group's base LR) for this step.
        if step < warm:
            return (step + 1) / warm  # rises linearly from 1/warm up to 1.0
        progress = (step - warm) / max(1, total - warm)  # 0 -> 1 over the cosine phase
        # cosine goes from 1.0 (progress 0) down to min_lr_ratio (progress 1); min() keeps it
        # at the floor if steps run past `total`.
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


def _mixup(images, labels, alpha):
    """Mixup: blend each image with another random image of the same batch.

    Inputs: images [B, 3, H, W], labels [B], alpha (Beta distribution parameter).
    Returns (mixed_images, labels_a, labels_b, lam). The loss is then
    lam * loss(labels_a) + (1 - lam) * loss(labels_b).
    """
    lam = float(np.random.beta(alpha, alpha))  # mixing weight in [0, 1]
    perm = torch.randperm(images.size(0), device=images.device)  # random partner per image
    return lam * images + (1 - lam) * images[perm], labels, labels[perm], lam


@torch.no_grad()  # no gradients needed for validation: saves memory and time
def _validate(model, loader, criterion, device):
    """Evaluate the model on `loader`.

    Returns (mean loss per sample, balanced accuracy, plain accuracy).
    """
    model.eval()  # eval mode: dropout off, batch-norm uses running statistics
    losses, ys, ps = [], [], []
    for batch in loader:
        x, y = batch["image"].to(device), batch["label"].to(device)
        logits = model(x)  # [B, num_classes]
        # criterion returns the batch mean, so multiply by batch size to get a sum
        losses.append(criterion(logits.float(), y).item() * len(y))
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())  # predicted class = highest score
    y, p = np.concatenate(ys), np.concatenate(ps)
    # total loss / number of samples = mean loss; (y == p).mean() = plain accuracy
    return sum(losses) / len(y), fast_balanced_accuracy(y, p), float((y == p).mean())


def train(cfg, loaders=None):
    """Train a model from `cfg`. `loaders` (dict train/val) can be injected, e.g. for tests.

    Returns a summary dict: best_val_bal_acc, best_epoch, checkpoint, log_csv, epochs_run.

    Steps: seed RNGs, build data/model/loss/optimizer/scheduler, then for each epoch train
    on all batches, validate, log a CSV row, save the checkpoint if val balanced accuracy
    improved, and stop early after `train.early_stopping_patience` epochs without
    improvement. The summary (plus "run") is also written to <run_dir>/summary.json.
    """
    seed_everything(cfg.get("seed", 42))
    # config "device" wins; otherwise use the GPU if there is one
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    t = cfg["train"]  # shortcut to the training sub-config
    name = run_name(cfg)
    out_dir = run_dir(cfg)  # <output_dir>/<run_name>: every run gets its own folder
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}_best.pt"

    loaders = loaders or get_dataloaders(cfg)  # use injected loaders if given
    train_loader, val_loader = loaders["train"], loaders["val"]
    # images per class in the training set (length NUM_CLASSES even if a class is missing)
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
    # len(train_loader) = batches per epoch, so the schedule is updated every step
    scheduler = build_scheduler(optimizer, t["epochs"], len(train_loader), t.get("warmup_epochs", 0))
    # AMP = automatic mixed precision (float16 where safe); only used on CUDA
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    # GradScaler scales the loss up so small float16 gradients do not underflow to zero;
    # with enabled=False it does nothing and training is normal float32.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Backbone freezing only applies to ResNet models; 0 means never frozen.
    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0) if cfg["model"]["name"].startswith("resnet") else 0
    grad_clip = t.get("grad_clip", 1.0)  # max gradient norm; 0/None disables clipping
    mixup_alpha = t.get("mixup_alpha", 0.0)  # 0 = mixup off

    # best val balanced acc so far, its epoch, epochs without improvement, per-epoch rows
    best, best_epoch, bad, history = -1.0, -1, 0, []
    for epoch in range(t["epochs"]):
        if freeze_epochs:
            # frozen (only fc trains) while epoch < freeze_epochs, then everything trains
            set_backbone_trainable(model, epoch >= freeze_epochs)
        model.train()  # training mode: dropout on, batch-norm updates its statistics
        t0, run_loss, seen = time.time(), 0.0, 0
        # progress bar over batches (hidden when train.progress is false)
        it = tqdm(train_loader, desc=f"epoch {epoch + 1}/{t['epochs']}", leave=False,
                  disable=not t.get("progress", True))
        for batch in it:
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)  # clear old gradients (None is cheaper than 0)
            with torch.autocast(device_type=device.type, enabled=use_amp):  # mixed precision
                if mixup_alpha > 0:
                    x, ya, yb, lam = _mixup(x, y, mixup_alpha)
                    logits = model(x)
                    # mixed target: weighted sum of the losses for both original labels;
                    # .float() computes the loss in float32 for stability under AMP
                    loss = lam * criterion(logits.float(), ya) + (1 - lam) * criterion(logits.float(), yb)
                else:
                    loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()  # backprop on the (possibly scaled) loss
            if grad_clip:
                scaler.unscale_(optimizer)  # undo scaling so clipping sees the true gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)  # optimizer step (skipped if gradients are inf/NaN)
            scaler.update()  # adjust the loss scale for the next step
            scheduler.step()  # LR schedule advances once per batch, not per epoch
            run_loss += loss.item() * len(y)  # sum of losses, to average over the epoch
            seen += len(y)

        val_loss, val_bal, val_acc = _validate(model, val_loader, criterion, device)
        # one log row per epoch; "lr" is the largest LR across param groups (the head's for ViT)
        row = {"epoch": epoch + 1, "train_loss": run_loss / seen, "val_loss": val_loss,
               "val_bal_acc": val_bal, "val_acc": val_acc, "lr": max(g["lr"] for g in optimizer.param_groups),
               "seconds": time.time() - t0}
        history.append(row)
        # rewrite the whole CSV every epoch so the log is up to date even if training crashes
        pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)
        log.info("epoch %d: %s", epoch + 1, {k: round(v, 4) for k, v in row.items()})

        if val_bal > best:  # new best model on validation balanced accuracy
            best, best_epoch, bad = val_bal, epoch + 1, 0  # reset the patience counter
            # save weights plus the config so the checkpoint can be rebuilt later
            torch.save({"model": model.state_dict(), "cfg": cfg, "run_name": name, "epoch": epoch + 1,
                        "val_bal_acc": val_bal},
                       ckpt_path)
        else:
            bad += 1  # one more epoch without improvement
            if bad >= t.get("early_stopping_patience", 5):  # default patience: 5 epochs
                log.info("early stopping at epoch %d (best %d)", epoch + 1, best_epoch)
                break

    summary = {"run": name, "best_val_bal_acc": best, "best_epoch": best_epoch,
               "epochs_run": len(history), "checkpoint": str(ckpt_path), "log_csv": str(out_dir / "log.csv")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
