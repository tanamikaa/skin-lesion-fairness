# This file tests the loss functions.
# Focal loss is cross entropy times an extra factor; with gamma = 0 that factor is 1,
# so focal loss must equal plain cross entropy.
# Run with: pytest tests/test_losses.py

# torch is used to make fake data
import torch

# the loss we are testing
from skinfair.losses import FocalLoss


# Test: with gamma = 0, FocalLoss gives the same value as cross entropy.
# Args: none. Returns: nothing (it fails if the assert is false).
def test_focal_equals_ce_when_gamma_zero():
    """Tests that focal loss equals normal cross entropy when gamma is 0.
    Step 1: make 8 fake images with 7 random class scores and random labels.
    Step 2: compute the built-in PyTorch cross entropy as the reference.
    Step 3: compute FocalLoss with gamma 0 and compare the two numbers.
    Input: none. Output: nothing, the test fails if they differ by more than 1e-6.
    It is needed to prove the focal loss maths is correct.
    """
    # 8 fake images, each with 7 random class scores
    logits = torch.randn(8, 7)
    # 8 random true labels between 0 and 6
    y = torch.randint(0, 7, (8,))
    # reference value from PyTorch's built-in cross entropy
    ce = torch.nn.functional.cross_entropy(logits, y)
    # the two numbers must match up to a tiny rounding error
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), ce, atol=1e-6)
