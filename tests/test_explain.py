import numpy as np
import torch

from skinfair.explain.attention import attention_rollout, gradcam_vit
from skinfair.explain.gradcam import gradcam
from skinfair.explain.quality import deletion_auc, insertion_auc, mask_iou, pointing_game, quality_by_group
from skinfair.models import build_model


def mk(name):
    torch.manual_seed(0)
    return build_model({"model": {"name": name, "pretrained": False, "num_classes": 7}}).eval()


def test_gradcam_shape_range_and_target():
    m = mk("resnet50")
    x = torch.randn(2, 3, 64, 64)
    maps = gradcam(m, x)
    assert maps.shape == (2, 64, 64) and maps.min() >= 0 and maps.max() <= 1
    assert not torch.allclose(gradcam(m, x, targets=[0, 0]), gradcam(m, x, targets=[5, 5]))
    assert not m.training


def test_attention_rollout_shape_range_and_restores_fused():
    m = mk("vit_base_patch16_224")
    flags = [b.attn.fused_attn for b in m.blocks]
    x = torch.randn(2, 3, 224, 224)
    maps = attention_rollout(m, x)
    assert maps.shape == (2, 224, 224)
    assert maps.min() >= 0 and maps.max() <= 1 + 1e-6 and maps.std() > 0
    assert [b.attn.fused_attn for b in m.blocks] == flags
    assert torch.allclose(m(x), m(x))  # model still usable, deterministic in eval


def test_rollout_without_discard_is_row_stochastic_effect():
    m = mk("vit_base_patch16_224")
    x = torch.randn(1, 3, 224, 224)
    a, b = attention_rollout(m, x, discard_ratio=0.0), attention_rollout(m, x, discard_ratio=0.9)
    assert a.shape == b.shape and not torch.allclose(a, b)


def test_gradcam_on_vit():
    m = mk("vit_base_patch16_224")
    maps = gradcam_vit(m, torch.randn(1, 3, 224, 224))
    assert maps.shape == (1, 224, 224) and maps.min() >= 0 and maps.max() <= 1


class BrightModel(torch.nn.Module):
    """Class 1 prob depends on mean intensity of the centre patch only (known ground truth)."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        s = x[:, :, 24:40, 24:40].mean(dim=(1, 2, 3)) * 10
        return torch.stack([torch.zeros_like(s), s], dim=1) + self.dummy


def test_deletion_insertion_prefer_the_faithful_map():
    m = BrightModel()
    x = -torch.ones(2, 3, 64, 64)
    x[:, :, 24:40, 24:40] = 1  # bright lesion-like centre on a dark background
    good = torch.zeros(2, 64, 64)
    good[:, 24:40, 24:40] = 1  # highlights the pixels the model uses
    bad = 1 - good             # highlights everything else
    tgt = [1, 1]
    assert (deletion_auc(m, x, good, targets=tgt) < deletion_auc(m, x, bad, targets=tgt)).all()
    assert (insertion_auc(m, x, good, targets=tgt) > insertion_auc(m, x, bad, targets=tgt)).all()


def test_pointing_game_iou_and_group_table():
    heat = torch.zeros(2, 8, 8)
    heat[0, 2, 2] = 1
    heat[1, 7, 7] = 1
    mask = torch.zeros(2, 8, 8)
    mask[:, :4, :4] = 1
    assert pointing_game(heat, mask).tolist() == [1.0, 0.0]
    assert mask_iou(heat, mask).tolist() == [1 / 16, 0.0]
    t = quality_by_group(np.array([0.2, 0.4, 0.9]), np.array([1, 1, -1]), "m")
    assert t["group"].tolist() == [1] and abs(t["value"][0] - 0.3) < 1e-9
