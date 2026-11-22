# This file is the shared training loop for both models (ResNet and ViT).
# It takes the data loaders and the model, teaches the model, and saves the best version.
# Features: mixed precision (CUDA only), gradient clipping, early stopping on validation
# balanced accuracy, best-checkpoint saving, a per-epoch CSV log, warm-up + cosine learning
# rate, layer-wise LR decay for ViT, backbone freezing for ResNet, and optional mixup.
# Words used here:
#   epoch = one full pass over the training images; batch = a small group of images;
#   step = one weight update (once per batch); loss = how wrong the model is;
#   logits = raw class scores; balanced accuracy = average recall over classes.
# It is called by scripts/train.py and scripts/hparam_search.py.

# Standard library tools: json for the summary, logging for messages, math for cosine,
# random for seeding, time for timing, Path for file paths.
import json
import logging
import math
import random
import time
from pathlib import Path

# numpy for arrays, pandas for the CSV log, torch for the neural network.
import numpy as np
import pandas as pd
import torch
# LambdaLR lets us change the learning rate with our own function.
from torch.optim.lr_scheduler import LambdaLR
# tqdm shows a progress bar.
from tqdm import tqdm

# Number of classes (7).
from skinfair import NUM_CLASSES
# Function that builds the train/val/test data loaders.
from skinfair.data.datasets import get_dataloaders
# Fast balanced accuracy used to score validation.
from skinfair.evaluation.metrics import fast_balanced_accuracy
# Builds the loss function from the config.
from skinfair.losses import build_loss
# Builds ResNet or ViT from the config.
from skinfair.models import build_model
# Freezes or unfreezes the ResNet backbone.
from skinfair.models.resnet import set_backbone_trainable
# Gives ViT layers different learning rates.
from skinfair.models.vit import layer_decay_param_groups
# Helpers for the run name and output folder.
from skinfair.utils import run_dir, run_name

# Logger for progress messages.
log = logging.getLogger(__name__)


# seed_everything: fixes all random seeds.
# Input: seed (int), e.g. 42.
# Returns: nothing.
# Why: weights start random and batches are shuffled; fixing the seed makes runs repeatable,
#   so differences between runs come from our changes, not luck.
def seed_everything(seed: int):
    """
    Fixes all random number generators to one seed value.
    Steps: it seeds Python's random, NumPy, PyTorch on CPU and PyTorch on every GPU.
    Input: seed, an integer such as 42.
    Output: nothing is returned.
    Weights start random and batches are shuffled, so this makes runs repeatable.
    Without it, differences between runs could just be luck.
    """
    # Python's own random numbers.
    random.seed(seed)
    # NumPy random numbers (also used by mixup).
    np.random.seed(seed)
    # PyTorch random numbers on CPU.
    torch.manual_seed(seed)
    # PyTorch random numbers on every GPU.
    torch.cuda.manual_seed_all(seed)


# build_optimizer: creates the AdamW optimizer.
# Inputs: model, and cfg (needs cfg["train"]["lr"], ["weight_decay"] and optionally
#   cfg["model"]["layer_decay"]).
# Returns: a torch.optim.AdamW object.
# Why: the optimizer changes the weights to reduce the loss. Weight decay gently pulls weights
#   towards zero to reduce overfitting. ViT can get a different learning rate per layer.
def build_optimizer(model, cfg):
    """
    Creates the AdamW optimizer that updates the model weights.
    Steps: read train and model settings, then check if this is a ViT with layer decay.
    If yes, it builds per-layer parameter groups; otherwise it uses one learning rate.
    Inputs: model and cfg (needs train lr, weight_decay, optional layer_decay).
    Output: a torch.optim.AdamW object.
    It is needed because the optimizer is what actually learns from the loss.
    """
    # Training settings from the config.
    t = cfg["train"]
    # Model settings from the config.
    m = cfg["model"]
    # ViT with layer_decay set: each layer gets its own learning rate.
    if m["name"].startswith("vit") and m.get("layer_decay"):
        # Make the parameter groups, each with its own lr and weight decay.
        groups = layer_decay_param_groups(model, t["lr"], t["weight_decay"], m["layer_decay"])
        # The groups already carry their own settings, so pass only the groups.
        return torch.optim.AdamW(groups)
    # Otherwise one learning rate for all parameters.
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


# build_scheduler: makes the learning-rate schedule (slow warm-up, then cosine decay).
# Inputs: optimizer (whose learning rate we change), epochs (total planned), steps_per_epoch
#   (batches per epoch), warmup_epochs (length of warm-up, may be fractional), min_lr_ratio
#   (final LR as a fraction of the start LR).
# Returns: a LambdaLR scheduler; call scheduler.step() after every optimizer step.
# Why: warm-up starts with a tiny LR so early steps do not wreck pretrained weights; the
#   cosine decay then makes big moves early and fine adjustments at the end.
def build_scheduler(optimizer, epochs, steps_per_epoch, warmup_epochs=0, min_lr_ratio=0.01):
    """
    Makes the learning-rate schedule: slow warm-up, then cosine decay.
    Steps: count total steps and warm-up steps, then define how the rate changes per step.
    Inputs: optimizer, epochs, steps_per_epoch, warmup_epochs and min_lr_ratio.
    Output: a LambdaLR scheduler; call its step() after every optimizer step.
    Warm-up protects pretrained weights early on.
    The decay makes big moves first and small careful moves at the end.
    """
    # Total number of optimizer steps (at least 1).
    total = max(1, epochs * steps_per_epoch)
    # Number of warm-up steps.
    warm = int(warmup_epochs * steps_per_epoch)

    # lr_lambda: gives the number we multiply the base learning rate by at a given step.
    # Input: step (int, the current step number).
    # Returns: a float between min_lr_ratio and 1.0.
    # Why: this is the shape of the schedule.
    def lr_lambda(step):
        """
        Gives the number that multiplies the base learning rate at one step.
        Steps: during warm-up it grows in a straight line up to 1.0.
        After warm-up it computes progress and follows a cosine curve downwards.
        Input: step, the current step number as an integer.
        Output: a float between min_lr_ratio and 1.0.
        It defines the shape of the whole learning-rate schedule.
        """
        # During warm-up.
        if step < warm:
            # Grow in a straight line up to 1.0.
            return (step + 1) / warm
        # How far we are in the cosine part, from 0 to 1.
        progress = (step - warm) / max(1, total - warm)
        # Cosine goes from 1.0 down to min_lr_ratio; min() stops it going past the end.
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # Wrap our function so PyTorch calls it every scheduler.step().
    return LambdaLR(optimizer, lr_lambda)


# _mixup: blends each image with another random image from the same batch.
# Inputs: images (tensor batch x 3 x H x W), labels (tensor of class numbers), alpha (shape
#   of the Beta distribution that picks the mixing amount; bigger = more mixing).
# Returns: (mixed_images, labels, shuffled_labels, lam) where lam is the mixing amount.
# Why: mixup is a regulariser; the model sees in-between images so it cannot just memorise.
def _mixup(images, labels, alpha):
    """
    Blends every image with another random image from the same batch.
    Steps: pick a mixing amount from a Beta distribution, then shuffle the batch order.
    Mix each image with its shuffled partner using that amount.
    Inputs: images, labels and alpha (bigger alpha means more mixing).
    Output: mixed images, original labels, shuffled labels and the mixing amount.
    It is a regulariser so the model cannot simply memorise the images.
    """
    # Random mixing amount between 0 and 1.
    lam = float(np.random.beta(alpha, alpha))
    # A random partner image for every image.
    perm = torch.randperm(images.size(0), device=images.device)
    # Return mixed images, both label sets, and the mixing amount.
    return lam * images + (1 - lam) * images[perm], labels, labels[perm], lam


# _validate: tests the model on a loader without changing the weights.
# Inputs: model, loader (validation DataLoader), criterion (loss function), device.
# Returns: tuple of three floats: mean loss, balanced accuracy, plain accuracy.
# Why: this score picks the best epoch and drives early stopping.
# No gradients are needed when validating, which saves memory and time.
@torch.no_grad()
def _validate(model, loader, criterion, device):
    """
    Tests the model on a data loader without changing any weights.
    Steps: switch to eval mode, loop over batches, get scores and compute the loss.
    It collects true labels and predictions for all batches.
    Inputs: model, loader, criterion (loss function) and device.
    Output: mean loss, balanced accuracy and plain accuracy as three floats.
    It is needed to pick the best epoch and to decide early stopping.
    """
    # Eval mode: dropout off, batch-norm uses its saved statistics.
    model.eval()
    # Lists to collect loss, true labels and predictions.
    losses, ys, ps = [], [], []
    # Go through the validation data batch by batch.
    for batch in loader:
        # Move the batch to the device (CPU or GPU).
        x, y = batch["image"].to(device), batch["label"].to(device)
        # Raw class scores from the model.
        logits = model(x)
        # The loss is a batch average, so times batch size to get a sum.
        losses.append(criterion(logits.float(), y).item() * len(y))
        # Keep true labels on the CPU as NumPy.
        ys.append(y.cpu().numpy())
        # Predicted class = the highest score.
        ps.append(logits.argmax(1).cpu().numpy())
    # Join all batches into one array each.
    y, p = np.concatenate(ys), np.concatenate(ps)
    # Mean loss, balanced accuracy, and plain accuracy.
    return sum(losses) / len(y), fast_balanced_accuracy(y, p), float((y == p).mean())


# train: trains a model from the config and returns a summary dict.
# Steps: seed and device, make folders, get data, build model/loss/optimizer/scheduler,
#   then for each epoch train on every batch, validate, log, and save the best checkpoint.
#   Stop early if validation does not improve for `patience` epochs.
# Inputs: cfg (config dict; keys used include seed, device, output_dir, checkpoint_dir,
#   train.*, model.*, loss.name, data.oversample), loaders (optional dict with "train"
#   and "val" DataLoaders, handy for tests).
# Returns: dict with run, best_val_bal_acc, best_epoch, epochs_run, checkpoint, log_csv.
# Why: one function trains both models under the same procedure so they are comparable.
#   The test set is never used here, so choosing the best epoch is honest.
def train(cfg, loaders=None):
    """
    Trains a model from a config and returns a short summary.
    Steps: seed and device, make folders, get data, build model, loss, optimizer and scheduler.
    Then each epoch trains on all batches, validates, logs, and saves the best checkpoint.
    It stops early if validation does not improve for several epochs.
    Inputs: cfg (config dict) and loaders (optional dict with train and val loaders).
    Output: dict with run name, best score, best epoch, epochs run, checkpoint and log paths.
    """
    # Fix the random seeds first.
    seed_everything(cfg.get("seed", 42))
    # Use the device from the config, or the GPU if there is one, else the CPU.
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Short name for the training settings.
    t = cfg["train"]
    # Name of this run.
    name = run_name(cfg)
    # Every run gets its own output folder.
    out_dir = run_dir(cfg)
    # Create the folder (and parents) if it does not exist.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Folder for the best saved model.
    ckpt_dir = Path(cfg.get("checkpoint_dir", "results/checkpoints"))
    # Create it if needed.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # File path of the best checkpoint.
    ckpt_path = ckpt_dir / f"{name}_best.pt"

    # Use the given loaders, or build them from the config.
    loaders = loaders or get_dataloaders(cfg)
    # Pick the train and validation loaders.
    train_loader, val_loader = loaders["train"], loaders["val"]
    # Number of training images per class (minlength keeps missing classes as 0);
    # the loss uses these counts to make class weights (rare class = bigger weight).
    counts = np.bincount(train_loader.dataset.labels, minlength=NUM_CLASSES)

    # Build the model and move it to the device.
    model = build_model(cfg).to(device)
    # Oversampling already balances classes, so class weights would correct twice.
    oversampled = bool(cfg.get("data", {}).get("oversample", False))
    # Warn when class weights will be switched off.
    if oversampled and cfg["loss"]["name"] in ("weighted_ce", "focal"):
        # Print the warning message.
        log.warning("oversample=true: class weights disabled for loss '%s' (avoid double correction)",
                    cfg["loss"]["name"])
    # Build the loss; class counts are given only when not oversampling.
    criterion = build_loss(cfg, class_counts=None if oversampled else counts).to(device)
    # Build the optimizer.
    optimizer = build_optimizer(model, cfg)
    # The schedule moves a little at every batch step.
    scheduler = build_scheduler(optimizer, t["epochs"], len(train_loader), t.get("warmup_epochs", 0))
    # Mixed precision (AMP, float16 maths = faster, less memory) only works on a CUDA GPU.
    use_amp = bool(t.get("amp", False)) and device.type == "cuda"
    # Scaler stops tiny float16 gradients from becoming zero; does nothing when disabled.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # Freezing the backbone is only for ResNet; 0 means never frozen.
    freeze_epochs = cfg["model"].get("freeze_backbone_epochs", 0) if cfg["model"]["name"].startswith("resnet") else 0
    # Biggest allowed gradient size (0 or None turns clipping off); gradient clipping stops
    # one bad batch from making a huge, harmful weight update.
    grad_clip = t.get("grad_clip", 1.0)
    # Mixup strength; 0 means off.
    mixup_alpha = t.get("mixup_alpha", 0.0)

    # Best score so far, its epoch, epochs without improvement, and the log rows.
    best, best_epoch, bad, history = -1.0, -1, 0, []
    # Loop over the epochs.
    for epoch in range(t["epochs"]):
        # Only when backbone freezing is used.
        if freeze_epochs:
            # Only the last layer trains until freeze_epochs is over, then all layers train.
            set_backbone_trainable(model, epoch >= freeze_epochs)
        # Training mode: dropout on, batch-norm updates its statistics.
        model.train()
        # Start time, running loss total, and number of samples seen.
        t0, run_loss, seen = time.time(), 0.0, 0
        # Progress bar over the batches (can be turned off in config).
        it = tqdm(train_loader, desc=f"epoch {epoch + 1}/{t['epochs']}", leave=False,
                  disable=not t.get("progress", True))
        # Loop over the batches.
        for batch in it:
            # Move images and labels to the device.
            x, y = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
            # Clear gradients from the last step (a gradient says how to change each weight).
            optimizer.zero_grad(set_to_none=True)
            # Run the forward pass in mixed precision if enabled.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # If mixup is on.
                if mixup_alpha > 0:
                    # Blend images and get both label sets.
                    x, ya, yb, lam = _mixup(x, y, mixup_alpha)
                    # Raw class scores for the blended images.
                    logits = model(x)
                    # Loss is the weighted mix of the loss for both labels (float32 is safer).
                    loss = lam * criterion(logits.float(), ya) + (1 - lam) * criterion(logits.float(), yb)
                # Otherwise no mixup.
                else:
                    # Normal loss.
                    loss = criterion(model(x).float(), y)
            # Backward pass: compute gradients (the loss is scaled up if using mixed precision).
            scaler.scale(loss).backward()
            # Only clip when a clip value is set.
            if grad_clip:
                # Undo the scaling so clipping sees the real gradients.
                scaler.unscale_(optimizer)
                # Shrink gradients if they are too big, to keep training stable.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Update the weights (skipped if gradients have inf or NaN).
            scaler.step(optimizer)
            # Adjust the scaler for the next step.
            scaler.update()
            # Move the learning-rate schedule one step (once per batch).
            scheduler.step()
            # Add this batch's total loss for the epoch average.
            run_loss += loss.item() * len(y)
            # Add this batch's sample count.
            seen += len(y)

        # Check the model on the validation set.
        val_loss, val_bal, val_acc = _validate(model, val_loader, criterion, device)
        # One row of numbers for this epoch; lr is the biggest learning rate among groups.
        row = {"epoch": epoch + 1, "train_loss": run_loss / seen, "val_loss": val_loss,
               "val_bal_acc": val_bal, "val_acc": val_acc, "lr": max(g["lr"] for g in optimizer.param_groups),
               "seconds": time.time() - t0}
        # Add the row to the history.
        history.append(row)
        # Rewrite the CSV every epoch so the log is safe even if training crashes.
        pd.DataFrame(history).to_csv(out_dir / "log.csv", index=False)
        # Print the row, rounded to 4 decimals.
        log.info("epoch %d: %s", epoch + 1, {k: round(v, 4) for k, v in row.items()})

        # Is this the best model so far (by validation balanced accuracy)?
        # A checkpoint is a saved copy of the model; we keep only the best epoch.
        if val_bal > best:
            # Remember it and reset the patience counter.
            best, best_epoch, bad = val_bal, epoch + 1, 0
            # Save weights and config so the model can be rebuilt later.
            torch.save({"model": model.state_dict(), "cfg": cfg, "run_name": name, "epoch": epoch + 1,
                        "val_bal_acc": val_bal},
                       ckpt_path)
        # Otherwise the model did not improve.
        else:
            # One more epoch with no improvement.
            bad += 1
            # Early stopping: stop when we waited too long (default 5 epochs), which saves
            # time and avoids overfitting.
            if bad >= t.get("early_stopping_patience", 5):
                # Tell the user why training stopped.
                log.info("early stopping at epoch %d (best %d)", epoch + 1, best_epoch)
                # Leave the epoch loop.
                break

    # Short result report of the run.
    summary = {"run": name, "best_val_bal_acc": best, "best_epoch": best_epoch,
               "epochs_run": len(history), "checkpoint": str(ckpt_path), "log_csv": str(out_dir / "log.csv")}
    # Save the report as a JSON file.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    # Give the report back to the caller.
    return summary
