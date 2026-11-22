"""ViT-B/16 fine-tuning. Owner: Khushi Narwariya."""
import timm
import torch.nn as nn


def build_vit(cfg):
    """Create a Vision Transformer with timm.

    Reads model.name (timm model name), model.pretrained (default True),
    model.num_classes and model.drop_path_rate (default 0.1). Returns the model.
    """
    m = cfg["model"]
    return timm.create_model(
        m["name"],
        pretrained=m.get("pretrained", True),
        num_classes=m["num_classes"],  # timm builds a new head with this many outputs
        # stochastic depth: randomly skips residual blocks in training (regularisation)
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


def _layer_id(name: str, num_layers: int) -> int:
    """0 = embeddings, 1..num_layers = transformer blocks, num_layers + 1 = final norm + head.

    Input: parameter name (e.g. "blocks.3.attn.qkv.weight") and number of blocks.
    Returns the layer index used to pick the LR scale.
    """
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1  # "blocks.3.xxx" -> block 3 -> layer id 4
    return num_layers + 1  # everything else (final norm, head) counts as the top layer


def _no_decay(name: str, p) -> bool:
    """Return True if this parameter should get no weight decay.

    That is: 1-D tensors (biases, LayerNorm weights), any ".bias", and the token /
    position embeddings. Decaying these usually hurts and does not regularise well.
    """
    return p.ndim <= 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed", "reg_token")


def layer_decay_param_groups(model, base_lr, weight_decay, layer_decay=0.75):
    """Layer-wise LR decay param groups for the optimizer (BEiT / timm recipe).

    The head gets `base_lr`; each block further from the head gets lr multiplied by
    `layer_decay` once more, so early layers change least. Biases, norms, cls_token and
    pos_embed get no weight decay. Every trainable parameter appears in exactly one group.

    Inputs: timm ViT model (must have .blocks), base LR, weight decay, decay factor.
    Returns a list of param-group dicts to pass to torch.optim.AdamW.
    Example with 12 blocks and 0.75: head scale 1.0, block 11 0.75, ..., embeddings 0.75^13.
    """
    num_layers = len(model.blocks)  # 12 for ViT-B/16
    # scales[i] = layer_decay^(distance from the top); scales[num_layers + 1] = 1.0 (head)
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    groups = {}  # (layer_id, decay?) -> param group
    for name, p in model.named_parameters():
        if not p.requires_grad:  # skip frozen parameters
            continue
        lid = _layer_id(name, num_layers)
        decay = not _no_decay(name, p)
        key = (lid, decay)  # one group per layer and per decay / no-decay
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * scales[lid],
                "lr_scale": scales[lid],  # extra info (checked in tests); AdamW ignores it
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def set_vit_backbone_trainable(model: nn.Module, trainable: bool):
    """Freeze (False) or unfreeze (True) every ViT parameter except the "head." layer.

    Changes the model in place; returns nothing.
    """
    for name, p in model.named_parameters():
        if not name.startswith("head."):  # timm names the classifier "head"
            p.requires_grad = trainable
