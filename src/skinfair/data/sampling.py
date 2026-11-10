"""Class-balanced oversampling. Owner: Afzal Raza."""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_balanced_sampler(labels):
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)
