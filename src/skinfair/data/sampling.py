"""Class-balanced oversampling. Owner: Afzal Raza."""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels, num_classes=None, seed=None):
    """Sampler that draws every class with equal probability (with replacement).

    Each sample gets weight 1/count(its class). `num_classes` makes the bincount robust to
    classes that are absent from `labels` (their count is never used).
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes or 0)
    weights = 1.0 / counts[labels]
    gen = None
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True, generator=gen
    )
