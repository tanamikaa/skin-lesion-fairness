"""Model factory: picks ResNet or ViT from the config, and loads saved checkpoints."""
import torch

from .resnet import build_resnet
from .vit import build_vit


def build_model(cfg):
    """Build the model named in cfg["model"]["name"].

    Names starting with "resnet" go to build_resnet, names starting with "vit" to build_vit
    (e.g. a timm name like "vit_base_patch16_224"). Returns an nn.Module.
    Raises ValueError for any other name.
    """
    name = cfg["model"]["name"]
    if name.startswith("resnet"):
        return build_resnet(cfg)
    if name.startswith("vit"):
        return build_vit(cfg)
    raise ValueError(f"Unknown model {name}")


def load_checkpoint(model, path, device="cpu"):
    """Load a checkpoint written by skinfair.train (dict with 'model' state_dict). Returns the ckpt dict.

    Inputs: a model with the same architecture, checkpoint path, device to load onto.
    The weights are copied into `model` in place.
    """
    # map_location puts tensors on `device` (e.g. GPU checkpoint on a CPU machine).
    # weights_only=False is needed because the file also holds the cfg dict and run info;
    # only load checkpoints you trust, since this can run pickled code.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])  # strict: parameter names and shapes must match
    return ckpt
