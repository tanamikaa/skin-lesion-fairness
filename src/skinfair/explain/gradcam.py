"""Grad-CAM for ResNet-50 (and, optionally, ViT). Owner: Anamika Kumari.

Interface: gradcam(model, images) -> Tensor[B, H, W] heat-maps in [0, 1].
"""
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def default_target_layer(model):
    """Last conv block of a torchvision ResNet.

    Grad-CAM uses this layer because it has the most high-level features while still
    keeping a (coarse) spatial layout. Returns the layer module.
    """
    return model.layer4[-1]


def gradcam(model, images, targets=None, target_layer=None, reshape_transform=None):
    """Grad-CAM heat-maps.

    Args:
        model: ResNet-50 (or any model, if `target_layer` is given).
        images: float Tensor[B, 3, H, W] (already normalised).
        targets: optional LongTensor/list[int] of class indices to explain; defaults to the
            model's predicted class for each image.
        target_layer: layer to hook; defaults to `model.layer4[-1]`.
        reshape_transform: needed for ViTs (see `explain.attention.vit_reshape_transform`).
    Returns:
        Tensor[B, H, W] on CPU, each map min-max normalised to [0, 1].

    How it works: Grad-CAM takes the gradient of the target class score with respect to the
    target layer's feature maps, uses the averaged gradient as a weight per channel, sums the
    weighted maps, keeps positive values and upsamples to image size.
    """
    target_layer = target_layer if target_layer is not None else default_target_layer(model)
    was_training = model.training
    model.eval()
    if targets is None:
        with torch.no_grad():
            targets = model(images).argmax(1)  # explain the predicted class of each image
    # Tell the library which output (class index) to explain for each image.
    cam_targets = [ClassifierOutputTarget(int(t)) for t in targets]
    # grad-cam's own min-max normalisation happens per image, so maps are in [0, 1].
    with GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        with torch.enable_grad():  # gradients are needed even if the caller disabled them
            maps = cam(input_tensor=images, targets=cam_targets)  # NumPy array [B, H, W]
    model.train(was_training)
    # Convert to a float tensor and clip to [0, 1] in place (guards against tiny overshoots).
    return torch.as_tensor(maps, dtype=torch.float32).clamp_(0, 1)
