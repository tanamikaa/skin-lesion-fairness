"""Loss functions for class imbalance. Owner: Anamika Kumari."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017). `alpha` = optional per-class weight tensor."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        logp_t = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def build_loss(cfg, class_counts=None):
    name = cfg["loss"]["name"]
    weights = None
    if class_counts is not None and name in ("weighted_ce", "focal"):
        counts = torch.as_tensor(class_counts, dtype=torch.float)
        weights = counts.sum() / (len(counts) * counts)
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(cfg["loss"].get("gamma", 2.0), weights)
    raise ValueError(f"Unknown loss {name}")
