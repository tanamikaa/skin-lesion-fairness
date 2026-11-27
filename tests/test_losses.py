import pytest
import torch

from skinfair.losses import FocalLoss, build_loss


def test_focal_equals_ce_when_gamma_zero():
    logits = torch.randn(8, 7)
    y = torch.randint(0, 7, (8,))
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, y), ce, atol=1e-6)


def test_focal_downweights_easy_examples():
    y = torch.tensor([0])
    easy, hard = torch.tensor([[10.0, 0, 0]]), torch.tensor([[0.0, 1.0, 0]])
    ce = torch.nn.functional.cross_entropy
    assert FocalLoss(2.0)(easy, y) / ce(easy, y) < FocalLoss(2.0)(hard, y) / ce(hard, y)


def test_build_loss_variants():
    counts = [100, 10, 1, 50, 5, 500, 20]
    logits, y = torch.randn(14, 7), torch.randint(0, 7, (14,))
    for name in ("ce", "weighted_ce", "focal"):
        loss = build_loss({"loss": {"name": name}}, counts)(logits, y)
        assert loss.ndim == 0 and torch.isfinite(loss)
    w = build_loss({"loss": {"name": "weighted_ce"}}, counts).weight
    assert w[2] > w[5]  # rare class weighted higher
    with pytest.raises(ValueError):
        build_loss({"loss": {"name": "nope"}})
