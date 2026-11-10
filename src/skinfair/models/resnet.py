"""ResNet-50 transfer-learning baseline. Owner: Anamika Kumari."""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet(cfg):
    m = cfg["model"]
    weights = ResNet50_Weights.IMAGENET1K_V2 if m.get("pretrained", True) else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, m["num_classes"])
    return model


def set_backbone_trainable(model, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable
