import torch

from skinfair.models import build_model, load_checkpoint
from skinfair.models.resnet import set_backbone_trainable
from skinfair.models.vit import layer_decay_param_groups


def cfg(name):
    return {"model": {"name": name, "pretrained": False, "num_classes": 7}}


def test_resnet_logits_shape_and_freeze():
    m = build_model(cfg("resnet50")).eval()
    assert m(torch.randn(2, 3, 64, 64)).shape == (2, 7)
    set_backbone_trainable(m, False)
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())


def test_vit_logits_shape():
    m = build_model(cfg("vit_base_patch16_224")).eval()
    assert m(torch.randn(2, 3, 224, 224)).shape == (2, 7)


def test_layer_decay_groups_cover_all_params_once_and_scale_monotonic():
    m = build_model(cfg("vit_base_patch16_224"))
    groups = layer_decay_param_groups(m, base_lr=1e-3, weight_decay=0.05, layer_decay=0.75)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))
    scales = sorted({g["lr_scale"] for g in groups})
    assert len(scales) == 14  # embeddings + 12 blocks + head
    assert scales[-1] == 1.0  # head trains at the base LR
    assert all(g["lr"] <= 1e-3 + 1e-12 for g in groups)
    assert min(g["lr"] for g in groups) < max(g["lr"] for g in groups)
    # no weight decay on biases / norms
    assert all(g["weight_decay"] == 0.0 for g in groups if all(p.ndim <= 1 for p in g["params"]))


def test_checkpoint_roundtrip(tmp_path):
    m = build_model(cfg("resnet50"))
    torch.save({"model": m.state_dict()}, tmp_path / "c.pt")
    m2 = build_model(cfg("resnet50"))
    load_checkpoint(m2, tmp_path / "c.pt")
    assert all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
