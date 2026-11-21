"""ViT-B/16 fine-tuning. Owner: Khushi Narwariya."""
import timm
import torch.nn as nn


def build_vit(cfg):
    m = cfg["model"]
    return timm.create_model(
        m["name"],
        pretrained=m.get("pretrained", True),
        num_classes=m["num_classes"],
        drop_path_rate=m.get("drop_path_rate", 0.1),
    )


def _layer_id(name: str, num_layers: int) -> int:
    """0 = embeddings, 1..num_layers = transformer blocks, num_layers + 1 = final norm + head."""
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers + 1


def _no_decay(name: str, p) -> bool:
    return p.ndim <= 1 or name.endswith(".bias") or name in ("cls_token", "pos_embed", "reg_token")


def layer_decay_param_groups(model, base_lr, weight_decay, layer_decay=0.75):
    """Layer-wise LR decay param groups for the optimizer (BEiT / timm recipe).

    The head gets `base_lr`; each block further from the head gets lr multiplied by
    `layer_decay` once more, so early layers change least. Biases, norms, cls_token and
    pos_embed get no weight decay. Every trainable parameter appears in exactly one group.
    """
    num_layers = len(model.blocks)
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name, num_layers)
        decay = not _no_decay(name, p)
        key = (lid, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * scales[lid],
                "lr_scale": scales[lid],
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(p)
    return list(groups.values())


def set_vit_backbone_trainable(model: nn.Module, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith("head."):
            p.requires_grad = trainable
