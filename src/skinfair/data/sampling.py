"""Class-balanced oversampling. Owner: Afzal Raza."""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels, num_classes=None, seed=None):
    """Sampler that draws every class with equal probability (with replacement).

    Each sample gets weight 1/count(its class). `num_classes` makes the bincount robust to
    classes that are absent from `labels` (their count is never used).

    Inputs: labels (int class per sample), num_classes, seed (for repeatable draws).
    Returns a WeightedRandomSampler that yields len(labels) indices per epoch.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes or 0)  # images per class
    # weight per sample = 1 / size of its class, so every class has the same total weight
    weights = 1.0 / counts[labels]
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    # replacement=True: rare-class images are drawn several times per epoch, and some
    # majority-class images may not be drawn at all in an epoch.
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True, generator=gen
    )
