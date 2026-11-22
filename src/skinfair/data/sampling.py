# This file is for oversampling during training.
# The HAM10000 data is imbalanced: one class ("nv") has most of the images.
# Oversampling shows rare-class images more often, so every class is seen about equally.
# Only the training loader uses it; validation and test data stay untouched.

# numpy is used for counting labels
import numpy as np
# torch is used for tensors and the random generator
import torch
# WeightedRandomSampler picks samples with chosen probabilities
from torch.utils.data import WeightedRandomSampler


# Builds a sampler that picks every class about equally often.
# Args:
#   labels: the class number of each training image.
#   num_classes: total number of classes, so missing classes are still counted.
#   seed: optional number that makes the random picks repeatable.
# Returns: a WeightedRandomSampler to give to a DataLoader instead of shuffle=True.
# Why: without it the model mostly sees "nv" and learns to ignore the rare classes.
def make_balanced_sampler(labels, num_classes=None, seed=None):
    """Builds a sampler that shows every class about equally often.
    Step 1: count how many training images each class has.
    Step 2: give each image a weight of 1 divided by its class size.
    Step 3: make a WeightedRandomSampler, with an optional seeded generator.
    Input: labels, num_classes (so empty classes are counted) and seed.
    Output: a sampler for the DataLoader. It stops the model ignoring rare classes.
    """
    # turn the labels into a numpy array of whole numbers
    labels = np.asarray(labels, dtype=np.int64)
    # count how many images each class has (minlength keeps classes with zero images)
    counts = np.bincount(labels, minlength=num_classes or 0)
    # each image gets weight 1 / (its class size), so every class has the same total weight
    weights = 1.0 / counts[labels]
    # start with no random generator
    gen = None
    # only make one if a seed was given
    if seed is not None:
        # a seeded generator makes the random picks repeatable
        gen = torch.Generator().manual_seed(seed)
    # return the sampler: weights as doubles, one draw per image (same epoch length),
    # replacement=True lets rare images be picked many times, and gen controls randomness
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True, generator=gen
    )
