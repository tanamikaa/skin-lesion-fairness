"""ViT-B/16 fine-tuning. Owner: Khushi Narwariya."""
import timm


def build_vit(cfg):
    m = cfg["model"]
    return timm.create_model(m["name"], pretrained=m.get("pretrained", True), num_classes=m["num_classes"])


def layer_decay_param_groups(model, base_lr, weight_decay, layer_decay=0.75):
    """TODO(Khushi): layer-wise LR decay param groups for the optimizer."""
    raise NotImplementedError
