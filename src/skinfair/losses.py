"""Loss functions for class imbalance. Owner: Anamika Kumari."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017). `alpha` = optional per-class weight tensor.

    Focal loss = -(1 - p_t)^gamma * log(p_t), where p_t is the predicted probability of the
    true class. The factor (1 - p_t)^gamma shrinks the loss of easy, already-correct samples,
    so training focuses on hard ones (often the rare classes). gamma = 0 gives plain CE.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        """Store gamma (focusing strength) and optional alpha (shape [num_classes])."""
        super().__init__()
        self.gamma = gamma
        # A buffer moves with .to(device) together with the module but is not trained.
        self.register_buffer("alpha", alpha)

    def forward(self, logits, target):
        """Compute the mean focal loss.

        Inputs: logits [batch, num_classes] (raw scores), target [batch] (class indices).
        Returns a scalar tensor.
        """
        # log-probs [B, C]; gather picks the column of the true class -> [B, 1] -> [B]
        logp_t = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t  # exp(log p_t) = p_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]  # weight each sample by its class weight
        return loss.mean()


def build_loss(cfg, class_counts=None):
    """Create the loss named in cfg["loss"]["name"]: "ce", "weighted_ce" or "focal".

    Inputs: config dict (reads loss.name and loss.gamma) and optional class_counts
    (number of training images per class). If class_counts is given, weighted_ce and focal
    use inverse-frequency class weights. Returns an nn.Module loss.
    """
    name = cfg["loss"]["name"]
    weights = None
    if class_counts is not None and name in ("weighted_ce", "focal"):
        counts = torch.as_tensor(class_counts, dtype=torch.float)
        # "balanced" weights: total / (num_classes * count_c). Rare classes get weight > 1,
        # common ones < 1; a perfectly balanced set gives weight 1 for every class.
        weights = counts.sum() / (len(counts) * counts)
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if name == "focal":
        return FocalLoss(cfg["loss"].get("gamma", 2.0), weights)
    raise ValueError(f"Unknown loss {name}")
