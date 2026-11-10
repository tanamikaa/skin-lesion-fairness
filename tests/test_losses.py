import torch

from skinfair.losses import FocalLoss


def test_focal_equals_ce_when_gamma_zero():
    logits = torch.randn(8, 7)
    y = torch.randint(0, 7, (8,))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), ce, atol=1e-6)
