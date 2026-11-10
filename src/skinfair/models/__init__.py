from .resnet import build_resnet
from .vit import build_vit


def build_model(cfg):
    name = cfg["model"]["name"]
    if name.startswith("resnet"):
        return build_resnet(cfg)
    if name.startswith("vit"):
        return build_vit(cfg)
    raise ValueError(f"Unknown model {name}")
