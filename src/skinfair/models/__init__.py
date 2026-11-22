# This file is the model "factory".
# It picks ResNet or ViT from the config and can load saved weights (checkpoints) into a model.
# build_model is used by training; load_checkpoint is used by evaluate and explain scripts.

# torch is needed to read the saved checkpoint file.
import torch

# The builder function for ResNet models.
from .resnet import build_resnet
# The builder function for ViT models.
from .vit import build_vit


# build_model: creates the model named in the config.
# Input: cfg, the config dict; needs cfg["model"]["name"] such as "resnet50" or "vit_base_patch16_224".
# Returns: a torch model that turns a batch of images into 7 class scores (logits).
# Why: the rest of the code does not need to know which model type is used.
def build_model(cfg):
    """
    Creates the model that is named in the config.
    Steps: read the model name, then choose the matching builder.
    Names starting with resnet use build_resnet; names starting with vit use build_vit.
    Any other name raises a clear ValueError.
    Input: cfg, the config dict; output: a torch model giving 7 class scores.
    The rest of the code does not need to know which model type is used.
    """
    # Read the model name from the config.
    name = cfg["model"]["name"]
    # Names starting with "resnet" use the ResNet builder.
    if name.startswith("resnet"):
        # Build and return the ResNet.
        return build_resnet(cfg)
    # Names starting with "vit" use the ViT builder.
    if name.startswith("vit"):
        # Build and return the ViT.
        return build_vit(cfg)
    # Any other name is not supported, so stop with a clear error.
    raise ValueError(f"Unknown model {name}")


# load_checkpoint: puts saved weights into a model.
# Inputs: model (built with the same config as in training), path (the .pt file),
#   device ("cpu" or "cuda") where the loaded numbers are placed.
# Returns: the whole checkpoint dict (weights, config, run name, epoch, score).
# Why: evaluation and explanation must use the weights of the best training epoch.
def load_checkpoint(model, path, device="cpu"):
    """
    Loads saved weights from a file into a model.
    Steps: read the file with torch.load on the chosen device.
    Then copy the saved weights into the model with load_state_dict.
    Inputs: model (built with the same config), path (.pt file) and device.
    Output: the whole checkpoint dict (weights, config, run name, epoch, score).
    Evaluation and explanation need the weights of the best training epoch.
    """
    # Read the file; map_location moves tensors to the device, weights_only=False is needed
    # because the file also stores the config (only open files you trust).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Copy the saved weights into the model (names and shapes must match).
    model.load_state_dict(ckpt["model"])
    # Give back the checkpoint so the caller can read its extra information.
    return ckpt
