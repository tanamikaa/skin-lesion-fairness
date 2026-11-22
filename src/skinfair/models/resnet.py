# This file builds the ResNet-50 model, the first (baseline) model.
# ResNet-50 is a CNN: a network that scans the image with small filters to find edges,
# textures and shapes.
# We use transfer learning: start from weights trained on ImageNet (a big set of everyday
# photos) so the model already knows basic patterns and learns skin lesions faster.

# nn holds the Linear layer
import torch.nn as nn
# the ResNet-50 model and its pretrained weights from torchvision
from torchvision.models import ResNet50_Weights, resnet50


# Builds a ResNet-50 with a new last layer for our classes.
# Args: cfg: config dict; uses cfg["model"]["num_classes"] (7) and optional
#       cfg["model"]["pretrained"] (default True).
# Returns: the ResNet-50; input [batch, 3, H, W], output logits [batch, num_classes].
# Why: ImageNet has 1000 classes but we have 7, so the old last layer is replaced.
def build_resnet(cfg):
    """Builds a ResNet-50 model for our 7 skin lesion classes.
    Step 1: read the model settings from the config.
    Step 2: load ImageNet weights, unless pretrained is false.
    Step 3: replace the last layer (fc) with a new one with num_classes outputs.
    Input: cfg, the config dict.
    Output: the model, taking [batch, 3, H, W] and giving logits. ImageNet has 1000 classes, we need 7.
    """
    # the model settings from the config
    m = cfg["model"]
    # use ImageNet weights unless pretrained is false (then start from random numbers)
    weights = ResNet50_Weights.IMAGENET1K_V2 if m.get("pretrained", True) else None
    # create the ResNet-50
    model = resnet50(weights=weights)
    # replace the last layer (fc) with a new one that has our number of outputs;
    # in_features is the size of the vector going into it (2048)
    model.fc = nn.Linear(model.fc.in_features, m["num_classes"])
    # give the model back
    return model


# Freezes or unfreezes every layer except the last one (fc).
# Args: model: the ResNet from build_resnet.
#       trainable: False = freeze the backbone, True = let it learn.
# Returns: nothing (the model is changed in place).
# Why: the new fc layer starts random and makes big errors; training the backbone at the
# same time could damage the good ImageNet weights. So we freeze first, then unfreeze.
def set_backbone_trainable(model, trainable: bool):
    """Freezes or unfreezes every layer of the ResNet except the last one.
    It goes through all parameters with their names.
    Any parameter that does not start with "fc." gets requires_grad set to trainable.
    The last layer (fc) is never changed, so it always learns.
    Input: model, and trainable (False to freeze, True to unfreeze).
    Output: nothing, the model changes in place. It protects the good ImageNet weights early on.
    """
    # go through every parameter together with its name
    for name, p in model.named_parameters():
        # the fc layer is always trained, so leave it alone
        if not name.startswith("fc."):
            # True = weights learn, False = frozen (no gradient, no update)
            p.requires_grad = trainable
