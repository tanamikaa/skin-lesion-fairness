import torch

from .resnet import build_resnet
from .vit import build_vit


def build_model(cfg):
    name = cfg["model"]["name"]
    if name.startswith("resnet"):
        return build_resnet(cfg)
    if name.startswith("vit"):
        return build_vit(cfg)
    raise ValueError(f"Unknown model {name}")


def load_checkpoint(model, path, device="cpu"):
    """Load a checkpoint written by skinfair.train (dict with 'model' state_dict). Returns the ckpt dict."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    return ckpt
