# This file has the loss functions used for training.
# A loss is one number that says how wrong the model is; training tries to make it small.
# HAM10000 is imbalanced, so a plain loss lets the model ignore rare classes.
# The losses here give more weight to rare or hard samples.

# torch is the main deep learning library
import torch
# nn holds building blocks like Module and CrossEntropyLoss
import torch.nn as nn
# F holds loss and math functions such as log_softmax
import torch.nn.functional as F


# Focal loss: a cross entropy loss where easy samples count less.
# Per sample: -(1 - p)^gamma * log(p), where p is the probability given to the true class.
# If the model is sure (p near 1) the sample barely counts; if unsure it counts a lot.
# gamma = 0 gives normal cross entropy. alpha is an optional weight for each class.
# Why: the many easy "nv" images stop dominating, so rare classes get learned.
class FocalLoss(nn.Module):
    # Stores the settings.
    # Args: gamma: how strongly to ignore easy samples (bigger = more).
    #       alpha: tensor of class weights with shape [num_classes], or None.
    # Returns: nothing (it builds the object).
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        """Sets up the focal loss with its two settings.
        It first calls the parent nn.Module setup.
        It saves gamma, which says how strongly easy samples are ignored.
        It saves alpha, the optional class weights, as a buffer.
        A buffer moves to the GPU with the model but is not trained.
        Input: gamma and alpha. Output: nothing, it only prepares the object.
        """
        # set up the parent nn.Module first
        super().__init__()
        # remember the focusing strength
        self.gamma = gamma
        # a buffer moves with the module to the GPU but is not trained
        self.register_buffer("alpha", alpha)

    # Computes the average focal loss for a batch (a group of images).
    # Args: logits: raw class scores, shape [batch, num_classes].
    #       target: true class numbers, shape [batch].
    # Returns: one number (the mean loss over the batch).
    def forward(self, logits, target):
        """Computes the average focal loss for one batch.
        Step 1: turn scores into log-probabilities and pick the true class of each image.
        Step 2: multiply by (1 - p) to the power gamma, so easy samples count less.
        Step 3: multiply by the class weight alpha if it exists.
        Input: logits [batch, classes] and target [batch].
        Output: one number, the mean loss. It helps rare classes get learned.
        """
        # turn scores into log-probabilities, pick the true class of each sample,
        # and remove the extra dimension
        logp_t = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        # exp() gives the probability p; (1 - p)^gamma is small for easy samples
        loss = -((1 - logp_t.exp()) ** self.gamma) * logp_t
        # only if class weights exist
        if self.alpha is not None:
            # multiply each sample's loss by the weight of its class
            loss = loss * self.alpha[target]
        # average over the batch
        return loss.mean()


# Creates the loss chosen in the config: "ce", "weighted_ce" or "focal".
# Args: cfg: config dict; uses cfg["loss"]["name"] and optionally cfg["loss"]["gamma"].
#       class_counts: number of training images per class, or None.
# Returns: a loss module, used as loss(logits, target).
# Why class_counts can be None: with oversampling the batches are already balanced,
# so weighting again would over-correct.
def build_loss(cfg, class_counts=None):
    """Creates the loss function chosen in the config.
    Step 1: read the loss name: ce, weighted_ce or focal.
    Step 2: if class counts exist and the loss is weighted, make inverse-frequency weights.
    Step 3: return the matching loss, or raise an error for an unknown name.
    Input: cfg, the config, and class_counts (can be None).
    Output: a loss module used as loss(logits, target). It keeps loss choice in the config.
    """
    # name of the loss the user asked for
    name = cfg["loss"]["name"]
    # by default there are no class weights
    weights = None
    # weights are made only for weighted losses and only if class sizes are known
    if class_counts is not None and name in ("weighted_ce", "focal"):
        # turn the counts into a float tensor
        counts = torch.as_tensor(class_counts, dtype=torch.float)
        # inverse-frequency weight: rare classes get big weights, common ones small
        weights = counts.sum() / (len(counts) * counts)
    # plain cross entropy
    if name == "ce":
        # no weights
        return nn.CrossEntropyLoss()
    # cross entropy with class weights
    if name == "weighted_ce":
        # pass the weights in
        return nn.CrossEntropyLoss(weight=weights)
    # focal loss
    if name == "focal":
        # gamma comes from the config (default 2.0), plus the class weights
        return FocalLoss(cfg["loss"].get("gamma", 2.0), weights)
    # any other name is a mistake in the config
    raise ValueError(f"Unknown loss {name}")
