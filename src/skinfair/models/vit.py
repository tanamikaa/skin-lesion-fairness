# This file builds the Vision Transformer (ViT-B/16), the second model, plus training helpers.
# A ViT cuts the image into 16x16-pixel patches and treats them like words in a sentence.
# Attention lets each patch look at all the others to decide what matters.
# We fine-tune it: start from pretrained weights and train a little more with small
# learning rates so we do not destroy what was already learned.

# timm is a library of ready-made models
import timm
# nn is used for the type hint in set_vit_backbone_trainable
import torch.nn as nn


# Creates a Vision Transformer with timm using settings from the config.
# Args: cfg: config dict; uses cfg["model"]["name"], ["num_classes"], and optional
#       ["pretrained"] (default True) and ["drop_path_rate"] (default 0.1).
# Returns: a timm ViT; input [batch, 3, 224, 224], output logits [batch, num_classes].
# Why: pretrained weights give a big head start because our dataset is small.
def build_vit(cfg):
    """Creates a Vision Transformer using the timm library.
    Step 1: read the model settings from the config.
    Step 2: ask timm to build the model with the given name.
    It loads pretrained weights unless the config says no, and sets drop_path_rate.
    It also makes a new last layer with num_classes outputs.
    Input: cfg. Output: the ViT model. Pretrained weights help because our data is small.
    """
    # the model settings from the config
    m = cfg["model"]
    # create and return the model; the first argument is the timm model name
    return timm.create_model(
        m["name"],
        # load pretrained weights unless the config says no
        pretrained=m.get("pretrained", True),
        # timm builds a new last layer with this many outputs
        num_classes=m["num_classes"],
        # drop_path randomly skips blocks while training, which helps against overfitting
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


# Gives a parameter a layer number: 0 = input side, 1..num_layers = blocks, last = head.
# Args: name: parameter name, e.g. "blocks.3.attn.qkv.weight".
#       num_layers: number of transformer blocks (12).
# Returns: an int from 0 to num_layers + 1.
# Why: layer-wise learning-rate decay needs to know how deep each parameter is.
def _layer_id(name: str, num_layers: int) -> int:
    """Gives a parameter a layer number from its name.
    Embedding parts (cls_token, pos_embed, patch_embed, reg_token) are layer 0.
    A name like "blocks.3.xxx" is layer 4, which is the block number plus 1.
    Everything else, like the final norm and the head, is the top layer.
    Input: name, the parameter name, and num_layers, the number of blocks.
    Output: a whole number. Layer-wise learning-rate decay needs it.
    """
    # embedding parts (they turn patches into numbers) are the lowest layer
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
        # layer 0
        return 0
    # a name like "blocks.3.xxx" belongs to block 3
    if name.startswith("blocks."):
        # block number plus 1, so blocks are layers 1..num_layers
        return int(name.split(".")[1]) + 1
    # everything else (final norm, head) is the top layer
    return num_layers + 1


# Decides if a parameter should get no weight decay.
# Args: name: parameter name; p: the parameter tensor.
# Returns: True if it should have no weight decay, else False.
# Why: weight decay pulls weights towards zero to fight overfitting, but it does not help
# on biases, normalisation weights and special tokens.
def _no_decay(name: str, p) -> bool:
    """Decides if a parameter should be left out of weight decay.
    It checks if the tensor is 1-D, which means biases and norm weights.
    It also checks if the name ends with .bias or is a special token.
    If any check is true, the answer is True.
    Input: name, the parameter name, and p, the parameter tensor.
    Output: True or False. Weight decay does not help these parameters.
    """
    # 1-D tensors are biases and norm weights; the names listed are special tokens
    return p.ndim <= 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed", "reg_token")


# Makes optimizer parameter groups where lower layers get a smaller learning rate.
# The head gets base_lr; each layer further down is multiplied by layer_decay once more.
# Args: model: the timm ViT.
#       base_lr: learning rate for the top layer (the head).
#       weight_decay: weight decay strength for parameters that use it.
#       layer_decay: factor between 0 and 1 (default 0.75).
# Returns: a list of dicts with "params", "lr", "lr_scale" and "weight_decay", ready for
#          torch.optim.AdamW.
# Why: early layers hold general knowledge (edges, textures) and should change slowly;
# the head is new and must learn fast.
def layer_decay_param_groups(model, base_lr, weight_decay, layer_decay=0.75):
    """Makes optimizer parameter groups where lower layers learn slower.
    Step 1: work out a learning-rate scale for every layer, 1.0 for the head.
    Step 2: go through trainable parameters and find each one layer and decay choice.
    Step 3: put parameters in groups with their own lr and weight decay.
    Input: model, base_lr, weight_decay and layer_decay (default 0.75).
    Output: a list of group dicts for AdamW. Early layers keep general knowledge, the head learns fast.
    """
    # number of transformer blocks (12 for ViT-B/16)
    num_layers = len(model.blocks)
    # learning-rate scale per layer: 1.0 for the head, smaller towards the input
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    # dict mapping (layer number, uses decay?) to a parameter group
    groups = {}
    # go through every parameter with its name
    for name, p in model.named_parameters():
        # frozen parameters are not trained, so skip them
        if not p.requires_grad:
            continue
        # the layer number of this parameter
        lid = _layer_id(name, num_layers)
        # True if this parameter should get weight decay
        decay = not _no_decay(name, p)
        # one group per layer and decay choice
        key = (lid, decay)
        # create the group the first time we see this key
        if key not in groups:
            # a new group dict
            groups[key] = {
                # start with an empty list of parameters
                "params": [],
                # this group's learning rate = base rate times its layer scale
                "lr": base_lr * scales[lid],
                # the scale itself, kept for tests; AdamW ignores it
                "lr_scale": scales[lid],
                # weight decay only if this group uses it
                "weight_decay": weight_decay if decay else 0.0,
            }
        # add this parameter to its group
        groups[key]["params"].append(p)
    # return just the groups as a list
    return list(groups.values())


# Freezes or unfreezes all ViT parameters except the head.
# Args: model: the timm ViT.
#       trainable: False = freeze, True = let it learn.
# Returns: nothing (the model is changed in place).
# Note: train.py does not call this; the ViT uses layer-wise LR decay instead.
def set_vit_backbone_trainable(model: nn.Module, trainable: bool):
    """Freezes or unfreezes all ViT parameters except the head.
    It goes through all parameters with their names.
    Any parameter that does not start with "head." gets requires_grad set to trainable.
    The head is the new classifier, so it is always trained.
    Input: model, and trainable (False to freeze, True to unfreeze).
    Output: nothing, the model changes in place. It lets us train the head first.
    """
    # go through every parameter with its name
    for name, p in model.named_parameters():
        # timm calls the classifier "head"; it is always trained
        if not name.startswith("head."):
            # True = weights learn, False = frozen
            p.requires_grad = trainable
