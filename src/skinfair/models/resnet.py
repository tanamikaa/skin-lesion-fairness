"""ResNet-50 transfer-learning baseline. Owner: Anamika Kumari."""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet(cfg):
    """Build a ResNet-50 with a new classification head.

    Reads model.pretrained (default True -> ImageNet weights) and model.num_classes.
    Returns the model; only the final fc layer is newly initialised.
    """
    m = cfg["model"]
    # IMAGENET1K_V2 = torchvision's improved ImageNet weights; None = random init
    weights = ResNet50_Weights.IMAGENET1K_V2 if m.get("pretrained", True) else None
    model = resnet50(weights=weights)
    # replace the 1000-class ImageNet head: in_features is 2048 for ResNet-50
    model.fc = nn.Linear(model.fc.in_features, m["num_classes"])
    return model


def set_backbone_trainable(model, trainable: bool):
    """Freeze (trainable=False) or unfreeze (True) every layer except the fc head.

    Frozen parameters get no gradients, so the optimizer does not change them.
    Returns nothing; the model is changed in place.
    """
    for name, p in model.named_parameters():
        if not name.startswith("fc."):  # "fc.weight" / "fc.bias" are the head; always train
            p.requires_grad = trainable
